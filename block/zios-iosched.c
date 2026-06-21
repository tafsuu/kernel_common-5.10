// SPDX-License-Identifier: GPL-2.0
/*
 * ZIOS I/O Scheduler v2.0.0 — Zenith I/O Scheduler
 *
 * A game-aware, top-app-cgroup-aware multi-queue I/O scheduler for
 * mobile devices (UFS / NVMe).  Designed for the Zenith kernel.
 *
 * Core ideas:
 *   - O(1) insert and dispatch (no rbtrees, no sorting, no RCU)
 *   - Game mode detection from the zenith CPU governor
 *   - Top-app cgroup awareness (foreground app I/O priority)
 *   - Read-first dispatch that becomes aggressive during gaming
 *   - Write absorption / batching during game mode
 *   - Sector-aware dispatch for better sequential throughput
 *   - Workload auto-detection (sequential, random, write-heavy)
 *   - Power-efficiency mode with auto battery detection
 *   - Writeback throttling (flusher threads at lowest priority)
 *   - Boot-time performance profile (aggressive flush first 30s)
 *   - Swap I/O detection and deprioritization
 *   - Thermal awareness (throttle I/O when device is hot)
 *   - Latency-sensitive process tracking (RT/FIFO priority)
 */
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/blkdev.h>
#include <linux/blk-mq.h>
#include <linux/elevator.h>
#include <linux/bio.h>
#include <linux/slab.h>
#include <linux/init.h>
#include <linux/jiffies.h>
#include <linux/sched.h>
#include <linux/sched/rt.h>
#include <linux/cgroup.h>
#include <linux/cpufreq_zenith.h>
#include <linux/power_supply.h>
#include <linux/thermal.h>
#include <linux/list_sort.h>

#include "blk.h"
#include "blk-mq.h"
#include "blk-mq-sched.h"
#include "blk-mq-debugfs.h"

#define ZIOS_VERSION "2.0.0"

/*
 * Debug logging gated behind CONFIG_ZENITH_DEBUG_MSG.
 * Production (community) builds compile these out entirely.
 */
#ifdef CONFIG_ZENITH_DEBUG_MSG
#define zios_debug(fmt, ...)	pr_debug("zios: " fmt, ##__VA_ARGS__)
#else
#define zios_debug(fmt, ...)	no_printk(KERN_DEBUG "zios: " fmt, ##__VA_ARGS__)
#endif

/* Per-request scheduler data */
struct zios_rq_data {
	struct request *rq;
	struct list_head top_app_node;	/* link in top_app_list if top-app */
};

/*
 * Per-queue scheduler data.
 */
struct zios_data {
	/* I/O lists */
	struct list_head	sync_read_list;
	struct list_head	async_read_list;
	struct list_head	write_list;
	struct list_head	writeback_list;
	struct list_head	swap_list;
	struct list_head	top_app_list;

	/* Counters */
	unsigned int		nr_sync_reads;
	unsigned int		nr_async_reads;
	unsigned int		nr_writes;
	unsigned int		nr_writeback;
	unsigned int		nr_swap;
	unsigned int		nr_top_app;

	/* Stats */
	atomic64_t		dispatched_reads;
	atomic64_t		dispatched_writes;
	atomic64_t		merged_requests;

	/* Last dispatched sector */
	sector_t		last_sector;
	bool			last_sector_valid;

	/* Dispatch accounting */
	unsigned int		dispensed;
	unsigned int		write_starve_count;
	unsigned int		writeback_starve_count;
	unsigned int		writeback_starve_max;
	bool			writeback_dirty;
	unsigned int		swap_starve_count;
	unsigned int		swap_starve_max;
	bool			swap_dirty;

	/* Game mode */
	bool			game_mode;
	unsigned long		last_game_check;

	/* Parameter sets */
	struct {
		unsigned int	read_batch_max;
		unsigned int	write_batch_max;
		unsigned int	top_app_bias_pct;
		unsigned int	write_starve_max;
	} game_params, normal_params, boot_params, thermal_params;

	unsigned int		read_coalesce_limit;
	unsigned int		max_write_pending;
	bool			writes_dirty;
	unsigned int		async_depth;
	char			top_app_cgroup_name[256];
	struct kmem_cache	*rq_data_cache;
	struct request_queue	*queue;
	spinlock_t		lock;

	/* Boot-time profile */
	bool			boot_mode;
	unsigned int		boot_duration_ms;
	unsigned long		boot_start;

	/* Power efficiency */
	bool			power_efficient;
	bool			auto_power_efficient;
	unsigned int		power_efficient_battery_pct;
	unsigned long		last_power_check;
	struct {
		unsigned int	read_batch_max;
		unsigned int	write_batch_max;
		unsigned int	top_app_bias_pct;
		unsigned int	write_starve_max;
	} power_efficient_params;

	/* Thermal throttling */
	bool			thermal_throttle;
	bool			auto_thermal_throttle;
	unsigned int		thermal_throttle_temp;
	char			thermal_zone_name[64];
	unsigned long		last_thermal_check;

	/* Latency-sensitive process tracking */
	bool			boost_rt_prio;
	bool			priority_inheritance;
	unsigned long		last_top_app_jiffies;

	/* Workload detection */
	bool			debug_log;
	unsigned int		sample_interval_ms;
	unsigned long		last_sample;
	u64			window_read_sectors;
	u64			window_write_sectors;
	u64			window_read_ops;
	u64			window_write_ops;
	sector_t		last_dispatch_sector;
	bool			last_dispatch_valid;
	u64			window_seq_ops;
	u64			window_rand_ops;
	unsigned int		window_peak_write_pending;
	enum {
		ZIOS_WORKLOAD_BALANCED	= 0,
		ZIOS_WORKLOAD_SEQUENTIAL,
		ZIOS_WORKLOAD_RANDOM,
		ZIOS_WORKLOAD_WRITE_HEAVY,
	} workload;
};

static int zios_to_word_depth(struct blk_mq_hw_ctx *hctx, unsigned int qdepth)
{
	struct sbitmap_queue *bt = hctx->sched_tags->bitmap_tags;
	const unsigned int nrr = hctx->queue->nr_requests;

	return ((qdepth << bt->sb.shift) + nrr - 1) / nrr;
}

static inline struct zios_rq_data *get_rq_data(struct request *rq)
{
	return rq->elv.priv[0];
}

/* ---------- cgroup helpers ---------- */

static inline bool task_in_cgroup_named(struct task_struct *task,
					const char *cgroup_name)
{
	struct cgroup *cgrp;
	bool ret = false;

	rcu_read_lock();
	cgrp = task_dfl_cgroup(task);

	while (cgrp) {
		if (cgrp->kn && !strcmp(cgrp->kn->name, cgroup_name)) {
			ret = true;
			goto out;
		}
		cgrp = cgroup_parent(cgrp);
	}
out:
	rcu_read_unlock();
	return ret;
}

/* ---------- LBA sort for write batching ---------- */

static int zios_cmp_write_lba(void *priv, struct list_head *a,
			      struct list_head *b)
{
	struct request *rq_a = list_entry(a, struct request, queuelist);
	struct request *rq_b = list_entry(b, struct request, queuelist);

	return (int)(blk_rq_pos(rq_a) > blk_rq_pos(rq_b)) -
	       (int)(blk_rq_pos(rq_a) < blk_rq_pos(rq_b));
}

static struct request *zios_find_nearest(struct list_head *list,
					 sector_t target,
					 unsigned int limit)
{
	struct request *rq, *best = NULL;
	sector_t best_delta = U64_MAX;
	unsigned int scanned = 0;

	list_for_each_entry(rq, list, queuelist) {
		sector_t delta;

		if (target > blk_rq_pos(rq))
			delta = target - blk_rq_pos(rq);
		else
			delta = blk_rq_pos(rq) - target;

		if (delta < best_delta) {
			best_delta = delta;
			best = rq;
		}

		if (++scanned >= limit)
			break;
	}

	return best;
}

/* ---------- request lifecycle ---------- */

static void zios_prepare_request(struct request *rq)
{
	struct zios_data *zd = rq->q->elevator->elevator_data;
	struct zios_rq_data *rd;

	rd = kmem_cache_alloc(zd->rq_data_cache, GFP_ATOMIC);
	if (!rd)
		return;

	memset(rd, 0, sizeof(*rd));
	rd->rq = rq;
	INIT_LIST_HEAD(&rd->top_app_node);
	rq->elv.priv[0] = rd;
}

static void zios_finish_request(struct request *rq)
{
	struct zios_data *zd = rq->q->elevator->elevator_data;
	struct zios_rq_data *rd = get_rq_data(rq);

	if (rd) {
		rq->elv.priv[0] = NULL;
		kmem_cache_free(zd->rq_data_cache, rd);
	}
}

/* Forward declarations */
static void zios_analyze_workload(struct zios_data *zd);
static void zios_check_thermal(struct zios_data *zd);

/* ---------- insert / classify ---------- */

static void zios_insert_request(struct blk_mq_hw_ctx *hctx,
				struct request *rq, bool at_head)
{
	struct request_queue *q = hctx->queue;
	struct zios_data *zd = q->elevator->elevator_data;
	struct zios_rq_data *rd = get_rq_data(rq);
	bool is_sync, is_write;
	bool is_top_app;

	lockdep_assert_held(&zd->lock);

	blk_req_zone_write_unlock(rq);

	if (!rd)
		return;

	if (blk_mq_sched_try_insert_merge(q, rq))
		return;

	blk_mq_sched_request_inserted(rq);

	is_sync = rq->cmd_flags & REQ_SYNC;
	is_write = op_is_write(req_op(rq));

	/* Check top-app cgroup */
	is_top_app = task_in_cgroup_named(current, zd->top_app_cgroup_name);

	/* Also boost RT/FIFO tasks if enabled (skip during boot to avoid I/O deadlocks) */
	if (!is_top_app && zd->boost_rt_prio && !zd->boot_mode) {
		if (rt_task(current))
			is_top_app = true;
	}

	/*
	 * Priority inheritance heuristic: if a top-app request was dispatched
	 * recently (within 50ms), boost I/O from other processes too.
	 * This catches helper threads doing I/O on behalf of the foreground app
	 * (content providers, binder threads, etc.).
	 */
	if (!is_top_app && zd->priority_inheritance &&
	    zd->last_top_app_jiffies &&
	    time_before(jiffies, zd->last_top_app_jiffies +
			msecs_to_jiffies(50))) {
		is_top_app = true;
	}

	if (at_head) {
		list_add(&rq->queuelist, &zd->sync_read_list);
		zd->nr_sync_reads++;
	} else if (is_top_app) {
		list_add_tail(&rd->top_app_node, &zd->top_app_list);
		list_add_tail(&rq->queuelist, &zd->top_app_list);
		zd->nr_top_app++;
	} else if (is_sync && !is_write) {
		list_add_tail(&rq->queuelist, &zd->sync_read_list);
		zd->nr_sync_reads++;
	} else if (!is_write) {
		list_add_tail(&rq->queuelist, &zd->async_read_list);
		zd->nr_async_reads++;
	} else if (rq->bio && (rq->bio->bi_opf & REQ_SWAP)) {
		/* Swap I/O — lowest priority */
		list_add_tail(&rq->queuelist, &zd->swap_list);
		zd->nr_swap++;
		zd->swap_dirty = true;
	} else if (current->flags & PF_SWAPWRITE) {
		/* Writeback (flusher thread) */
		list_add_tail(&rq->queuelist, &zd->writeback_list);
		zd->nr_writeback++;
		zd->writeback_dirty = true;
	} else {
		list_add_tail(&rq->queuelist, &zd->write_list);
		zd->nr_writes++;
		zd->writes_dirty = true;
	}
}

static void zios_insert_requests(struct blk_mq_hw_ctx *hctx,
				 struct list_head *list, bool at_head)
{
	struct zios_data *zd = hctx->queue->elevator->elevator_data;

	spin_lock(&zd->lock);
	while (!list_empty(list)) {
		struct request *rq;

		rq = list_first_entry(list, struct request, queuelist);
		list_del_init(&rq->queuelist);
		zios_insert_request(hctx, rq, at_head);
	}
	spin_unlock(&zd->lock);
}

/* ---------- thermal check ---------- */

static void zios_check_thermal(struct zios_data *zd)
{
	struct thermal_zone_device *tz;
	int temp, ret;

	if (time_before(jiffies, zd->last_thermal_check + 30 * HZ))
		return;
	zd->last_thermal_check = jiffies;

	tz = thermal_zone_get_zone_by_name(zd->thermal_zone_name);
	if (IS_ERR(tz)) {
		pr_warn_once("zios: thermal zone '%s' not found, thermal throttling disabled\n",
			     zd->thermal_zone_name);
		return;
	}

	ret = thermal_zone_get_temp(tz, &temp);
	if (ret)
		return;

	if (temp >= (int)zd->thermal_throttle_temp * 1000 && !zd->thermal_throttle) {
		zd->thermal_throttle = true;
		zios_debug("thermal: %d mC >= %u C -> throttle ON\n",
			   temp, zd->thermal_throttle_temp);
	} else if (temp < (int)(zd->thermal_throttle_temp - 5) * 1000 &&
		   zd->thermal_throttle) {
		zd->thermal_throttle = false;
		zios_debug("thermal: %d mC < %u C -> throttle OFF\n",
			   temp, zd->thermal_throttle_temp - 5);
	}
}

/* ---------- power efficiency ---------- */

static void zios_check_battery(struct zios_data *zd)
{
	struct power_supply *psy;
	union power_supply_propval val;
	int ret;

	if (time_before(jiffies, zd->last_power_check + 60 * HZ))
		return;
	zd->last_power_check = jiffies;

	psy = power_supply_get_by_name("battery");
	if (IS_ERR_OR_NULL(psy))
		return;

	ret = power_supply_get_property(psy, POWER_SUPPLY_PROP_CAPACITY, &val);
	if (ret == 0) {
		bool low = val.intval <= (int)zd->power_efficient_battery_pct;
		bool on_ac;

		on_ac = power_supply_is_system_supplied() > 0;

		if (!on_ac && low && !zd->power_efficient) {
			zd->power_efficient = true;
			zios_debug("battery %d%% <= %u%% -> power efficient ON\n",
				   val.intval, zd->power_efficient_battery_pct);
		} else if ((on_ac || !low) && zd->power_efficient) {
			zd->power_efficient = false;
			zios_debug("battery %d%% > %u%% or AC -> power efficient OFF\n",
				   val.intval, zd->power_efficient_battery_pct);
		}
	} else if (zd->power_efficient) {
		zd->power_efficient = false;
	}

	power_supply_put(psy);
}

/* ---------- dispatch ---------- */

static struct request *zios_dispatch_request(struct blk_mq_hw_ctx *hctx)
{
	struct zios_data *zd = hctx->queue->elevator->elevator_data;
	struct request *rq = NULL;
	bool game_mode;
	unsigned int rbatch, wbatch, bias_pct, starve_max;
	unsigned int coalesce;
	bool boot_mode;

	if (time_after(jiffies, zd->last_game_check + HZ / 10)) {
		zd->game_mode = zenith_is_game_mode_active();
		zd->last_game_check = jiffies;
	}
	game_mode = zd->game_mode;
	boot_mode = zd->boot_mode;

	if (zd->auto_power_efficient &&
	    time_after(jiffies, zd->last_power_check + 60 * HZ))
		zios_check_battery(zd);

	if (zd->auto_thermal_throttle &&
	    time_after(jiffies, zd->last_thermal_check + 30 * HZ))
		zios_check_thermal(zd);

	spin_lock(&zd->lock);

	if (game_mode) {
		rbatch    = zd->game_params.read_batch_max;
		wbatch    = zd->game_params.write_batch_max;
		bias_pct  = zd->game_params.top_app_bias_pct;
		starve_max = zd->game_params.write_starve_max;
	} else if (boot_mode) {
		rbatch    = zd->boot_params.read_batch_max;
		wbatch    = zd->boot_params.write_batch_max;
		bias_pct  = zd->boot_params.top_app_bias_pct;
		starve_max = zd->boot_params.write_starve_max;
	} else if (zd->thermal_throttle) {
		rbatch    = zd->thermal_params.read_batch_max;
		wbatch    = zd->thermal_params.write_batch_max;
		bias_pct  = zd->thermal_params.top_app_bias_pct;
		starve_max = zd->thermal_params.write_starve_max;
	} else if (zd->power_efficient) {
		rbatch    = zd->power_efficient_params.read_batch_max;
		wbatch    = zd->power_efficient_params.write_batch_max;
		bias_pct  = zd->power_efficient_params.top_app_bias_pct;
		starve_max = zd->power_efficient_params.write_starve_max;
	} else {
		rbatch    = zd->normal_params.read_batch_max;
		wbatch    = zd->normal_params.write_batch_max;
		bias_pct  = zd->normal_params.top_app_bias_pct;
		starve_max = zd->normal_params.write_starve_max;
	}
	coalesce = zd->power_efficient ?
		min(zd->read_coalesce_limit, 4u) :
		zd->read_coalesce_limit;

	/* Tier 0: Top-app */
	if (zd->nr_top_app > 0) {
		unsigned int reserve = (zd->dispensed * bias_pct) / 100;

		if (zd->nr_top_app > reserve) {
			struct zios_rq_data *rd;

			rd = list_first_entry(&zd->top_app_list,
					      struct zios_rq_data, top_app_node);
			rq = rd->rq;
			list_del_init(&rd->top_app_node);
			list_del_init(&rq->queuelist);
			zd->nr_top_app--;

			if (zd->nr_writes > 0)
				zd->write_starve_count++;

			/* Track for priority inheritance */
			zd->last_top_app_jiffies = jiffies;

			goto done;
		}
	}

	/* Tier 1: Sync reads */
	if (zd->nr_sync_reads > 0) {
		if (game_mode || zd->write_starve_count < starve_max) {
			if (coalesce > 1 && zd->last_sector_valid)
				rq = zios_find_nearest(&zd->sync_read_list,
						       zd->last_sector, coalesce);
			else
				rq = list_first_entry_or_null(&zd->sync_read_list,
						struct request, queuelist);

			if (rq) {
				list_del_init(&rq->queuelist);
				zd->nr_sync_reads--;
				if (zd->nr_writes > 0)
					zd->write_starve_count++;
				goto done;
			}
		}
	}

	/* Tier 2: Async reads */
	if (zd->nr_async_reads > 0) {
		if (game_mode || zd->write_starve_count < starve_max) {
			if (coalesce > 1 && zd->last_sector_valid)
				rq = zios_find_nearest(&zd->async_read_list,
						       zd->last_sector, coalesce);
			else
				rq = list_first_entry_or_null(&zd->async_read_list,
						struct request, queuelist);

			if (rq) {
				list_del_init(&rq->queuelist);
				zd->nr_async_reads--;
				if (zd->nr_writes > 0)
					zd->write_starve_count++;
				goto done;
			}
		}
	}

	/* Tier 3: Regular writes */
	if (zd->nr_writes > 0) {
		if (zd->write_starve_count >= starve_max ||
		    zd->nr_writes >= wbatch ||
		    zd->nr_writes >= zd->max_write_pending ||
		    (!zd->nr_sync_reads && !zd->nr_async_reads && !zd->nr_top_app)) {

			if (zd->writes_dirty && zd->nr_writes > 1) {
				list_sort(NULL, &zd->write_list,
					  zios_cmp_write_lba);
				zd->writes_dirty = false;
			}

			rq = list_first_entry_or_null(&zd->write_list,
						struct request, queuelist);
			if (rq) {
				list_del_init(&rq->queuelist);
				zd->nr_writes--;
				zd->write_starve_count = 0;
				goto done;
			}
		}
	}

	/* Tier 4: Writeback */
	if (zd->nr_writeback > 0) {
		if (zd->writeback_starve_count >= zd->writeback_starve_max ||
		    (!zd->nr_sync_reads && !zd->nr_async_reads &&
		     !zd->nr_writes && !zd->nr_top_app && !zd->nr_swap)) {

			if (zd->writeback_dirty && zd->nr_writeback > 1) {
				list_sort(NULL, &zd->writeback_list,
					  zios_cmp_write_lba);
				zd->writeback_dirty = false;
			}

			rq = list_first_entry_or_null(&zd->writeback_list,
						struct request, queuelist);
			if (rq) {
				list_del_init(&rq->queuelist);
				zd->nr_writeback--;
				zd->writeback_starve_count = 0;
				goto done;
			}
		}
		zd->writeback_starve_count++;
	}

	/* Tier 5: Swap I/O */
	if (zd->nr_swap > 0) {
		if (zd->swap_starve_count >= zd->swap_starve_max ||
		    (!zd->nr_sync_reads && !zd->nr_async_reads &&
		     !zd->nr_writes && !zd->nr_top_app && !zd->nr_writeback)) {

			if (zd->swap_dirty && zd->nr_swap > 1) {
				list_sort(NULL, &zd->swap_list,
					  zios_cmp_write_lba);
				zd->swap_dirty = false;
			}

			rq = list_first_entry_or_null(&zd->swap_list,
						struct request, queuelist);
			if (rq) {
				list_del_init(&rq->queuelist);
				zd->nr_swap--;
				zd->swap_starve_count = 0;
				goto done;
			}
		}
		zd->swap_starve_count++;
	}

done:
	if (rq) {
		zd->dispensed++;
		if (zd->dispensed > 256)
			zd->dispensed = 0;

		zd->last_sector = blk_rq_pos(rq) + blk_rq_sectors(rq);
		zd->last_sector_valid = true;

		if (op_is_write(req_op(rq))) {
			atomic64_inc(&zd->dispatched_writes);
			zd->window_write_sectors += blk_rq_sectors(rq);
			zd->window_write_ops++;
		} else {
			atomic64_inc(&zd->dispatched_reads);
			zd->window_read_sectors += blk_rq_sectors(rq);
			zd->window_read_ops++;
		}

		if (zd->last_dispatch_valid) {
			sector_t pos = blk_rq_pos(rq);
			sector_t last = zd->last_dispatch_sector;
			sector_t delta = (pos > last) ? (pos - last) : (last - pos);

			if (delta <= 32)
				zd->window_seq_ops++;
			else
				zd->window_rand_ops++;
		}
		zd->last_dispatch_sector = blk_rq_pos(rq) + blk_rq_sectors(rq);
		zd->last_dispatch_valid = true;

		if (zd->nr_writes > zd->window_peak_write_pending)
			zd->window_peak_write_pending = zd->nr_writes;

		/* Check boot mode expiry */
		if (zd->boot_mode && time_after(jiffies, zd->boot_start +
						msecs_to_jiffies(zd->boot_duration_ms))) {
			zd->boot_mode = false;
			zios_debug("boot mode finished after %u ms\n",
				   zd->boot_duration_ms);
		}

		if (!zd->game_mode &&
		    time_after(jiffies, zd->last_sample +
			       msecs_to_jiffies(zd->sample_interval_ms))) {
			zios_analyze_workload(zd);
			zd->last_sample = jiffies;
		}

		blk_req_zone_write_lock(rq);
		rq->rq_flags |= RQF_STARTED;
		spin_unlock(&zd->lock);
		return rq;
	}

	spin_unlock(&zd->lock);
	return NULL;
}

/* ---------- has work / depth limit ---------- */

static bool zios_has_work(struct blk_mq_hw_ctx *hctx)
{
	struct zios_data *zd = hctx->queue->elevator->elevator_data;

	return !list_empty_careful(&zd->sync_read_list) ||
	       !list_empty_careful(&zd->async_read_list) ||
	       !list_empty_careful(&zd->write_list) ||
	       !list_empty_careful(&zd->writeback_list) ||
	       !list_empty_careful(&zd->swap_list) ||
	       !list_empty_careful(&zd->top_app_list);
}

static void zios_limit_depth(unsigned int op, struct blk_mq_alloc_data *data)
{
	struct zios_data *zd = data->q->elevator->elevator_data;
	unsigned int depth;

	if (op_is_sync(op) && !op_is_write(op))
		return;

	depth = zd->async_depth;
	if (zd->power_efficient)
		depth = min(depth, zd->async_depth / 2);
	if (zd->thermal_throttle)
		depth = min(depth, zd->async_depth / 2);

	data->shallow_depth = zios_to_word_depth(data->hctx, depth);
}

static void zios_depth_updated(struct blk_mq_hw_ctx *hctx)
{
	struct request_queue *q = hctx->queue;
	struct zios_data *zd = q->elevator->elevator_data;
	struct blk_mq_tags *tags = hctx->sched_tags;

	zd->async_depth = q->nr_requests;
	sbitmap_queue_min_shallow_depth(tags->bitmap_tags, 1);
}

static int zios_init_hctx(struct blk_mq_hw_ctx *hctx, unsigned int hctx_idx)
{
	zios_depth_updated(hctx);
	return 0;
}

/* ---------- merge & completion ---------- */

static bool zios_bio_merge(struct request_queue *q, struct bio *bio,
			   unsigned int nr_segs)
{
	struct zios_data *zd = q->elevator->elevator_data;
	struct request *free = NULL;
	bool ret;

	spin_lock(&zd->lock);
	ret = blk_mq_sched_try_merge(q, bio, nr_segs, &free);
	spin_unlock(&zd->lock);

	if (free)
		blk_mq_free_request(free);

	return ret;
}

static void zios_request_merged(struct request_queue *q, struct request *req,
				enum elv_merge type)
{
}

static void zios_merged_requests(struct request_queue *q, struct request *req,
				 struct request *next)
{
	struct zios_data *zd = q->elevator->elevator_data;

	lockdep_assert_held(&zd->lock);
	atomic64_inc(&zd->merged_requests);
	zios_finish_request(next);
}

static void zios_completed_request(struct request *rq, u64 now)
{
}

/* ---------- workload analysis ---------- */

static void zios_analyze_workload(struct zios_data *zd)
{
	u64 total_read_sz, total_write_sz;
	u64 total_ops, read_ops, write_ops;
	unsigned int seq_ratio, rand_ratio;
	unsigned int write_ratio;
	unsigned int avg_read_sz_kb, avg_write_sz_kb;

	total_read_sz  = zd->window_read_sectors;
	total_write_sz = zd->window_write_sectors;
	read_ops  = zd->window_read_ops;
	write_ops = zd->window_write_ops;
	total_ops = read_ops + write_ops;

	if (total_ops < 8)
		goto reset_window;

	seq_ratio  = zd->window_seq_ops * 100 / total_ops;
	rand_ratio = zd->window_rand_ops * 100 / total_ops;
	write_ratio = write_ops * 100 / total_ops;

	avg_read_sz_kb  = read_ops  ? (total_read_sz  * 512 / 1024 / read_ops)  : 0;
	avg_write_sz_kb = write_ops ? (total_write_sz * 512 / 1024 / write_ops) : 0;

	zios_debug("workload: seq=%u%% rand=%u%% write=%u%% "
		   "avg_r=%uKB avg_w=%uKB peak_w=%u\n",
		   seq_ratio, rand_ratio, write_ratio,
		   avg_read_sz_kb, avg_write_sz_kb,
		   zd->window_peak_write_pending);

	if (seq_ratio > 70 && write_ratio < 40) {
		zd->workload = ZIOS_WORKLOAD_SEQUENTIAL;
		zd->read_coalesce_limit = 24;
		zd->normal_params.read_batch_max = 32;
		/* Increase read-ahead for sequential throughput */
		zd->queue->backing_dev_info->ra_pages = 256 >> (PAGE_SHIFT - 10);
		zios_debug("-> SEQUENTIAL (coalesce=%u, rbatch=%u, ra=%luKB)\n",
			   zd->read_coalesce_limit,
			   zd->normal_params.read_batch_max,
			   zd->queue->backing_dev_info->ra_pages << (PAGE_SHIFT - 10));
	} else if (rand_ratio > 60 && read_ops > write_ops) {
		zd->workload = ZIOS_WORKLOAD_RANDOM;
		zd->read_coalesce_limit = 4;
		zd->normal_params.read_batch_max = 8;
		zd->normal_params.write_starve_max = 5;
		/* Decrease read-ahead for random I/O (save memory/cache) */
		zd->queue->backing_dev_info->ra_pages = 32 >> (PAGE_SHIFT - 10);
		zios_debug("-> RANDOM (coalesce=%u, rbatch=%u, starve=%u, ra=%luKB)\n",
			   zd->read_coalesce_limit,
			   zd->normal_params.read_batch_max,
			   zd->normal_params.write_starve_max,
			   zd->queue->backing_dev_info->ra_pages << (PAGE_SHIFT - 10));
	} else if (write_ratio > 55 && total_write_sz > 0) {
		zd->workload = ZIOS_WORKLOAD_WRITE_HEAVY;
		zd->normal_params.write_batch_max = 32;
		zd->normal_params.write_starve_max = 2;
		zd->max_write_pending = 128;
		zios_debug("-> WRITE_HEAVY (wbatch=%u, starve=%u, maxw=%u)\n",
			   zd->normal_params.write_batch_max,
			   zd->normal_params.write_starve_max,
			   zd->max_write_pending);
	} else {
		if (zd->workload != ZIOS_WORKLOAD_BALANCED) {
			zd->read_coalesce_limit = 8;
			zd->normal_params.read_batch_max = 16;
			zd->normal_params.write_batch_max = 16;
			zd->normal_params.write_starve_max = 3;
			zd->max_write_pending = 64;
			/* Restore default read-ahead */
			zd->queue->backing_dev_info->ra_pages = 128 >> (PAGE_SHIFT - 10);
			zios_debug("-> BALANCED (restored defaults, ra=%luKB)\n",
				   zd->queue->backing_dev_info->ra_pages << (PAGE_SHIFT - 10));
		}
		zd->workload = ZIOS_WORKLOAD_BALANCED;
	}

reset_window:
	zd->window_read_sectors  = 0;
	zd->window_write_sectors = 0;
	zd->window_read_ops      = 0;
	zd->window_write_ops     = 0;
	zd->window_seq_ops       = 0;
	zd->window_rand_ops      = 0;
	zd->window_peak_write_pending = 0;
	zd->last_dispatch_valid  = false;
}

/* ---------- init / exit ---------- */

static void zios_exit_sched(struct elevator_queue *e)
{
	struct zios_data *zd = e->elevator_data;

	WARN_ON_ONCE(!list_empty(&zd->sync_read_list));
	WARN_ON_ONCE(!list_empty(&zd->async_read_list));
	WARN_ON_ONCE(!list_empty(&zd->write_list));
	WARN_ON_ONCE(!list_empty(&zd->writeback_list));
	WARN_ON_ONCE(!list_empty(&zd->swap_list));
	WARN_ON_ONCE(!list_empty(&zd->top_app_list));

	kmem_cache_destroy(zd->rq_data_cache);
	kfree(zd);
}

static int zios_init_sched(struct request_queue *q, struct elevator_type *e)
{
	struct zios_data *zd;
	struct elevator_queue *eq;

	eq = elevator_alloc(q, e);
	if (!eq)
		return -ENOMEM;

	zd = kzalloc_node(sizeof(*zd), GFP_KERNEL, q->node);
	if (!zd) {
		kobject_put(&eq->kobj);
		return -ENOMEM;
	}

	zd->rq_data_cache = kmem_cache_create("zios_rq_data",
					      sizeof(struct zios_rq_data),
					      0, SLAB_HWCACHE_ALIGN, NULL);
	if (!zd->rq_data_cache) {
		kfree(zd);
		kobject_put(&eq->kobj);
		return -ENOMEM;
	}

	INIT_LIST_HEAD(&zd->sync_read_list);
	INIT_LIST_HEAD(&zd->async_read_list);
	INIT_LIST_HEAD(&zd->write_list);
	INIT_LIST_HEAD(&zd->writeback_list);
	INIT_LIST_HEAD(&zd->swap_list);
	INIT_LIST_HEAD(&zd->top_app_list);

	zd->game_params.read_batch_max   = 64;
	zd->game_params.write_batch_max  = 4;
	zd->game_params.top_app_bias_pct = 80;
	zd->game_params.write_starve_max = 10;

	zd->normal_params.read_batch_max   = 16;
	zd->normal_params.write_batch_max  = 16;
	zd->normal_params.top_app_bias_pct = 50;
	zd->normal_params.write_starve_max = 3;

	zd->boot_params.read_batch_max   = 24;
	zd->boot_params.write_batch_max  = 4;
	zd->boot_params.top_app_bias_pct = 60;
	zd->boot_params.write_starve_max = 1;

	zd->thermal_params.read_batch_max   = 8;
	zd->thermal_params.write_batch_max  = 24;
	zd->thermal_params.top_app_bias_pct = 30;
	zd->thermal_params.write_starve_max = 10;

	zd->read_coalesce_limit = 8;
	zd->max_write_pending = 64;
	zd->async_depth = q->nr_requests;

	strscpy(zd->top_app_cgroup_name, "top-app",
		sizeof(zd->top_app_cgroup_name));

	zd->last_game_check = jiffies;
	zd->game_mode = false;
	zd->last_sector_valid = false;
	zd->writes_dirty = false;

	zd->nr_writeback = 0;
	zd->writeback_starve_count = 0;
	zd->writeback_starve_max = 20;
	zd->writeback_dirty = false;

	zd->nr_swap = 0;
	zd->swap_starve_count = 0;
	zd->swap_starve_max = 30;
	zd->swap_dirty = false;

	zd->boot_mode = true;
	zd->boot_duration_ms = 30000;
	zd->boot_start = jiffies;

	zd->power_efficient = false;
	zd->auto_power_efficient = true;
	zd->power_efficient_battery_pct = 15;
	zd->last_power_check = jiffies;
	zd->power_efficient_params.read_batch_max   = 8;
	zd->power_efficient_params.write_batch_max  = 24;
	zd->power_efficient_params.top_app_bias_pct = 30;
	zd->power_efficient_params.write_starve_max = 8;

	zd->thermal_throttle = false;
	zd->auto_thermal_throttle = true;
	zd->thermal_throttle_temp = 55;
	strscpy(zd->thermal_zone_name, "cpu-thermal",
		sizeof(zd->thermal_zone_name));
	zd->last_thermal_check = jiffies;

	zd->boost_rt_prio = false;
	zd->priority_inheritance = true;
	zd->last_top_app_jiffies = 0;

	zd->debug_log = false;
	zd->sample_interval_ms = 500;
	zd->last_sample = jiffies;
	zd->window_read_sectors = 0;
	zd->window_write_sectors = 0;
	zd->window_read_ops = 0;
	zd->window_write_ops = 0;
	zd->window_seq_ops = 0;
	zd->window_rand_ops = 0;
	zd->window_peak_write_pending = 0;
	zd->last_dispatch_sector = 0;
	zd->last_dispatch_valid = false;
	zd->workload = ZIOS_WORKLOAD_BALANCED;

	spin_lock_init(&zd->lock);
	zd->queue = q;
	eq->elevator_data = zd;
	q->elevator = eq;

	return 0;
}

/* ---------- sysfs attributes ---------- */

#define ZIOS_ATTR_RW_SHOW(name, field)					\
static ssize_t zios_##name##_show(struct elevator_queue *e, char *page)	\
{									\
	struct zios_data *zd = e->elevator_data;			\
	return sysfs_emit(page, "%u\n", zd->field);			\
}

#define ZIOS_ATTR_RW_STORE(name, field, min, max)			\
static ssize_t zios_##name##_store(struct elevator_queue *e,		\
				   const char *page, size_t count)	\
{									\
	struct zios_data *zd = e->elevator_data;			\
	unsigned int val;						\
	int ret;							\
	ret = kstrtouint(page, 10, &val);				\
	if (ret || val < (min) || val > (max))				\
		return -EINVAL;						\
	zd->field = val;						\
	return count;							\
}

#define ZIOS_RO(name, field) \
	ZIOS_ATTR_RW_SHOW(name, field)

#define ZIOS_RW(name, field, min, max)		\
	ZIOS_ATTR_RW_SHOW(name, field)		\
	ZIOS_ATTR_RW_STORE(name, field, min, max)

ZIOS_RW(game_read_batch_max,     game_params.read_batch_max,     1, 256)
ZIOS_RW(game_write_batch_max,    game_params.write_batch_max,    1, 256)
ZIOS_RW(game_top_app_bias_pct,   game_params.top_app_bias_pct,   0, 100)
ZIOS_RW(game_write_starve_max,   game_params.write_starve_max,   1, 100)

ZIOS_RW(normal_read_batch_max,   normal_params.read_batch_max,   1, 256)
ZIOS_RW(normal_write_batch_max,  normal_params.write_batch_max,  1, 256)
ZIOS_RW(normal_top_app_bias_pct, normal_params.top_app_bias_pct, 0, 100)
ZIOS_RW(normal_write_starve_max, normal_params.write_starve_max, 1, 100)

ZIOS_RW(boot_read_batch_max,     boot_params.read_batch_max,     1, 256)
ZIOS_RW(boot_write_batch_max,    boot_params.write_batch_max,    1, 256)
ZIOS_RW(boot_top_app_bias_pct,   boot_params.top_app_bias_pct,   0, 100)
ZIOS_RW(boot_write_starve_max,   boot_params.write_starve_max,   1, 100)

ZIOS_RW(thermal_read_batch_max,     thermal_params.read_batch_max,     1, 256)
ZIOS_RW(thermal_write_batch_max,    thermal_params.write_batch_max,    1, 256)
ZIOS_RW(thermal_top_app_bias_pct,   thermal_params.top_app_bias_pct,   0, 100)
ZIOS_RW(thermal_write_starve_max,   thermal_params.write_starve_max,   1, 100)

ZIOS_RW(power_efficient_read_batch_max,  power_efficient_params.read_batch_max,   1, 256)
ZIOS_RW(power_efficient_write_batch_max, power_efficient_params.write_batch_max,  1, 256)
ZIOS_RW(power_efficient_top_app_bias_pct,power_efficient_params.top_app_bias_pct, 0, 100)
ZIOS_RW(power_efficient_write_starve_max,power_efficient_params.write_starve_max, 1, 100)

ZIOS_RO(game_mode, game_mode)
ZIOS_RO(nr_sync_reads,  nr_sync_reads)
ZIOS_RO(nr_async_reads, nr_async_reads)
ZIOS_RO(nr_writes, nr_writes)
ZIOS_RO(nr_writeback, nr_writeback)
ZIOS_RO(nr_swap, nr_swap)
ZIOS_RO(nr_top_app, nr_top_app)
ZIOS_RO(boot_mode, boot_mode)
ZIOS_RO(thermal_throttle, thermal_throttle)

ZIOS_RW(read_coalesce_limit, read_coalesce_limit, 1, 64)
ZIOS_RW(max_write_pending, max_write_pending, 1, 4096)
ZIOS_RW(async_depth, async_depth, 1, INT_MAX)

ZIOS_RW(writeback_starve_max, writeback_starve_max, 1, 100)
ZIOS_RW(swap_starve_max, swap_starve_max, 1, 100)

ZIOS_RW(boot_duration_ms, boot_duration_ms, 1000, 120000)

ZIOS_RW(power_efficient_battery_pct, power_efficient_battery_pct, 1, 100)
ZIOS_RW(thermal_throttle_temp, thermal_throttle_temp, 30, 100)

/* Thermal zone name (string) */
static ssize_t zios_thermal_zone_name_show(struct elevator_queue *e, char *page)
{
	struct zios_data *zd = e->elevator_data;
	return sysfs_emit(page, "%s\n", zd->thermal_zone_name);
}

static ssize_t zios_thermal_zone_name_store(struct elevator_queue *e,
					    const char *page, size_t count)
{
	struct zios_data *zd = e->elevator_data;
	char buf[64];
	int ret;

	ret = sscanf(page, "%63s", buf);
	if (ret != 1)
		return -EINVAL;

	strscpy(zd->thermal_zone_name, buf,
		sizeof(zd->thermal_zone_name));
	return count;
}

/* Version */
static ssize_t zios_version_show(struct elevator_queue *e, char *page)
{
	return sysfs_emit(page, "%s\n", ZIOS_VERSION);
}

/* Stats */
static ssize_t zios_dispatched_reads_show(struct elevator_queue *e, char *page)
{
	struct zios_data *zd = e->elevator_data;
	return sysfs_emit(page, "%llu\n", atomic64_read(&zd->dispatched_reads));
}

static ssize_t zios_dispatched_writes_show(struct elevator_queue *e, char *page)
{
	struct zios_data *zd = e->elevator_data;
	return sysfs_emit(page, "%llu\n", atomic64_read(&zd->dispatched_writes));
}

static ssize_t zios_merged_requests_show(struct elevator_queue *e, char *page)
{
	struct zios_data *zd = e->elevator_data;
	return sysfs_emit(page, "%llu\n", atomic64_read(&zd->merged_requests));
}

/* Workload profile */
static ssize_t zios_workload_show(struct elevator_queue *e, char *page)
{
	struct zios_data *zd = e->elevator_data;
	const char *profile;

	switch (zd->workload) {
	case ZIOS_WORKLOAD_SEQUENTIAL:
		profile = "sequential";
		break;
	case ZIOS_WORKLOAD_RANDOM:
		profile = "random";
		break;
	case ZIOS_WORKLOAD_WRITE_HEAVY:
		profile = "write_heavy";
		break;
	default:
		profile = "balanced";
		break;
	}

	return sysfs_emit(page, "%s\n", profile);
}

/* Power efficiency toggle */
static ssize_t zios_power_efficient_show(struct elevator_queue *e, char *page)
{
	struct zios_data *zd = e->elevator_data;
	return sysfs_emit(page, "%d\n", zd->power_efficient);
}

static ssize_t zios_power_efficient_store(struct elevator_queue *e,
					  const char *page, size_t count)
{
	struct zios_data *zd = e->elevator_data;
	bool val;
	int ret;

	ret = kstrtobool(page, &val);
	if (ret)
		return -EINVAL;

	zd->power_efficient = val;
	return count;
}

/* Auto power efficiency */
static ssize_t zios_auto_power_efficient_show(struct elevator_queue *e, char *page)
{
	struct zios_data *zd = e->elevator_data;
	return sysfs_emit(page, "%d\n", zd->auto_power_efficient);
}

static ssize_t zios_auto_power_efficient_store(struct elevator_queue *e,
					       const char *page, size_t count)
{
	struct zios_data *zd = e->elevator_data;
	bool val;
	int ret;

	ret = kstrtobool(page, &val);
	if (ret)
		return -EINVAL;

	zd->auto_power_efficient = val;
	return count;
}

/* Auto thermal throttle */
static ssize_t zios_auto_thermal_throttle_show(struct elevator_queue *e, char *page)
{
	struct zios_data *zd = e->elevator_data;
	return sysfs_emit(page, "%d\n", zd->auto_thermal_throttle);
}

static ssize_t zios_auto_thermal_throttle_store(struct elevator_queue *e,
						const char *page, size_t count)
{
	struct zios_data *zd = e->elevator_data;
	bool val;
	int ret;

	ret = kstrtobool(page, &val);
	if (ret)
		return -EINVAL;

	zd->auto_thermal_throttle = val;
	return count;
}	/* Priority inheritance toggle */
static ssize_t zios_priority_inheritance_show(struct elevator_queue *e, char *page)
{
	struct zios_data *zd = e->elevator_data;
	return sysfs_emit(page, "%d\n", zd->priority_inheritance);
}

static ssize_t zios_priority_inheritance_store(struct elevator_queue *e,
						const char *page, size_t count)
{
	struct zios_data *zd = e->elevator_data;
	bool val;
	int ret;

	ret = kstrtobool(page, &val);
	if (ret)
		return -EINVAL;

	zd->priority_inheritance = val;
	return count;
}

/* Read-ahead KB (shows/sets the backing_dev_info ra_pages, rounded) */
static ssize_t zios_read_ahead_kb_show(struct elevator_queue *e, char *page)
{
	struct zios_data *zd = e->elevator_data;
	unsigned long ra_kb = zd->queue->backing_dev_info->ra_pages <<
				(PAGE_SHIFT - 10);
	return sysfs_emit(page, "%lu\n", ra_kb);
}

static ssize_t zios_read_ahead_kb_store(struct elevator_queue *e,
					 const char *page, size_t count)
{
	struct zios_data *zd = e->elevator_data;
	unsigned long ra_kb;
	int ret;

	ret = kstrtoul(page, 10, &ra_kb);
	if (ret || ra_kb > 4096)
		return -EINVAL;

	zd->queue->backing_dev_info->ra_pages = ra_kb >> (PAGE_SHIFT - 10);
	return count;
}

/* Boost RT/FIFO */
static ssize_t zios_boost_rt_prio_show(struct elevator_queue *e, char *page)
{
	struct zios_data *zd = e->elevator_data;
	return sysfs_emit(page, "%d\n", zd->boost_rt_prio);
}

static ssize_t zios_boost_rt_prio_store(struct elevator_queue *e,
					const char *page, size_t count)
{
	struct zios_data *zd = e->elevator_data;
	bool val;
	int ret;

	ret = kstrtobool(page, &val);
	if (ret)
		return -EINVAL;

	zd->boost_rt_prio = val;
	return count;
}

/* Debug log toggle (engineering builds only) */
#ifdef CONFIG_ZENITH_DEBUG_MSG
static ssize_t zios_debug_log_show(struct elevator_queue *e, char *page)
{
	struct zios_data *zd = e->elevator_data;
	return sysfs_emit(page, "%d\n", zd->debug_log);
}

static ssize_t zios_debug_log_store(struct elevator_queue *e,
				    const char *page, size_t count)
{
	struct zios_data *zd = e->elevator_data;
	bool val;
	int ret;

	ret = kstrtobool(page, &val);
	if (ret)
		return -EINVAL;

	zd->debug_log = val;
	return count;
}
#else
static ssize_t zios_debug_log_show(struct elevator_queue *e, char *page)
{
	return sysfs_emit(page, "0\n");
}

static ssize_t zios_debug_log_store(struct elevator_queue *e,
				    const char *page, size_t count)
{
	return count;
}
#endif

/* Top-app cgroup name */
static ssize_t zios_top_app_cgroup_name_show(struct elevator_queue *e,
					     char *page)
{
	struct zios_data *zd = e->elevator_data;
	return sysfs_emit(page, "%s\n", zd->top_app_cgroup_name);
}

static ssize_t zios_top_app_cgroup_name_store(struct elevator_queue *e,
					      const char *page, size_t count)
{
	struct zios_data *zd = e->elevator_data;
	char buf[256];
	int ret;

	ret = sscanf(page, "%255s", buf);
	if (ret != 1)
		return -EINVAL;

	strscpy(zd->top_app_cgroup_name, buf,
		sizeof(zd->top_app_cgroup_name));
	return count;
}

#define ZIOS_ATTR(name) __ATTR(name, 0644, zios_##name##_show, zios_##name##_store)

static struct elv_fs_entry zios_attrs[] = {
	ZIOS_ATTR(game_read_batch_max),
	ZIOS_ATTR(game_write_batch_max),
	ZIOS_ATTR(game_top_app_bias_pct),
	ZIOS_ATTR(game_write_starve_max),

	ZIOS_ATTR(normal_read_batch_max),
	ZIOS_ATTR(normal_write_batch_max),
	ZIOS_ATTR(normal_top_app_bias_pct),
	ZIOS_ATTR(normal_write_starve_max),

	ZIOS_ATTR(boot_read_batch_max),
	ZIOS_ATTR(boot_write_batch_max),
	ZIOS_ATTR(boot_top_app_bias_pct),
	ZIOS_ATTR(boot_write_starve_max),

	ZIOS_ATTR(thermal_read_batch_max),
	ZIOS_ATTR(thermal_write_batch_max),
	ZIOS_ATTR(thermal_top_app_bias_pct),
	ZIOS_ATTR(thermal_write_starve_max),

	ZIOS_ATTR(power_efficient),
	ZIOS_ATTR(auto_power_efficient),
	ZIOS_ATTR(power_efficient_battery_pct),
	ZIOS_ATTR(power_efficient_read_batch_max),
	ZIOS_ATTR(power_efficient_write_batch_max),
	ZIOS_ATTR(power_efficient_top_app_bias_pct),
	ZIOS_ATTR(power_efficient_write_starve_max),

	__ATTR(game_mode, 0444, zios_game_mode_show, NULL),
	__ATTR(nr_sync_reads, 0444, zios_nr_sync_reads_show, NULL),
	__ATTR(nr_async_reads, 0444, zios_nr_async_reads_show, NULL),
	__ATTR(nr_writes, 0444, zios_nr_writes_show, NULL),
	__ATTR(nr_writeback, 0444, zios_nr_writeback_show, NULL),
	__ATTR(nr_swap, 0444, zios_nr_swap_show, NULL),
	__ATTR(nr_top_app, 0444, zios_nr_top_app_show, NULL),
	__ATTR(boot_mode, 0444, zios_boot_mode_show, NULL),
	__ATTR(thermal_throttle, 0444, zios_thermal_throttle_show, NULL),

	ZIOS_ATTR(read_coalesce_limit),
	ZIOS_ATTR(max_write_pending),
	ZIOS_ATTR(writeback_starve_max),
	ZIOS_ATTR(swap_starve_max),

	ZIOS_ATTR(boot_duration_ms),
	ZIOS_ATTR(thermal_throttle_temp),
	ZIOS_ATTR(thermal_zone_name),
	ZIOS_ATTR(auto_thermal_throttle),
	ZIOS_ATTR(boost_rt_prio),
	ZIOS_ATTR(priority_inheritance),
	ZIOS_ATTR(read_ahead_kb),
	ZIOS_ATTR(async_depth),
	ZIOS_ATTR(top_app_cgroup_name),

	__ATTR(workload, 0444, zios_workload_show, NULL),
	ZIOS_ATTR(debug_log),

	__ATTR(dispatched_reads, 0444, zios_dispatched_reads_show, NULL),
	__ATTR(dispatched_writes, 0444, zios_dispatched_writes_show, NULL),
	__ATTR(merged_requests, 0444, zios_merged_requests_show, NULL),

	__ATTR(version, 0444, zios_version_show, NULL),

	__ATTR_NULL
};

/* ---------- elevator type ---------- */

static struct elevator_type mq_zios = {
	.ops = {
		.depth_updated		= zios_depth_updated,
		.limit_depth		= zios_limit_depth,
		.insert_requests	= zios_insert_requests,
		.dispatch_request	= zios_dispatch_request,
		.prepare_request	= zios_prepare_request,
		.finish_request		= zios_finish_request,
		.completed_request	= zios_completed_request,
		.bio_merge		= zios_bio_merge,
		.request_merged		= zios_request_merged,
		.requests_merged	= zios_merged_requests,
		.has_work		= zios_has_work,
		.init_sched		= zios_init_sched,
		.exit_sched		= zios_exit_sched,
		.init_hctx		= zios_init_hctx,
		.next_request		= elv_rb_latter_request,
		.former_request		= elv_rb_former_request,
	},

	.elevator_attrs = zios_attrs,
	.elevator_name = "zios",
	.elevator_owner = THIS_MODULE,
};
MODULE_ALIAS("mq-zios-iosched");

static int __init zios_init(void)
{
	pr_info("ZIOS I/O Scheduler v%s loaded\n", ZIOS_VERSION);
	return elv_register(&mq_zios);
}

static void __exit zios_exit(void)
{
	elv_unregister(&mq_zios);
}

module_init(zios_init);
module_exit(zios_exit);

MODULE_AUTHOR("Zenith Kernel");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("ZIOS - Zenith I/O Scheduler, game-aware, top-app-aware, boot-optimized, thermal-aware");
