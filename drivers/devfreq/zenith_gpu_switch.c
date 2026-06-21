// SPDX-License-Identifier: GPL-2.0-only
/*
 * Generic GPU devfreq governor auto-switcher for the zenith governor.
 *
 * Watches for zenith game-mode transitions via zenith_is_game_mode_active()
 * and automatically switches the GPU's devfreq governor:
 *   - game mode active  -> "performance"
 *   - game mode stopped -> "simple_ondemand" (with configurable idle timeout
 *     before falling to "simple_ondemand")
 *
 * Works with ANY GPU driver (Mali, Adreno, Panfrost, etc.) that registers
 * a devfreq device, not just Qualcomm's msm_gpu.
 *
 * Tunables are exposed under /sys/module/zenith_gpu_switch/parameters/
 */

#include <linux/module.h>
#include <linux/devfreq.h>
#include <linux/workqueue.h>
#include <linux/device.h>
#include <linux/cpufreq_zenith.h>
#include <linux/jiffies.h>
#include <linux/string.h>
#include <linux/slab.h>
#include <linux/compaction.h>
#include <linux/dcache.h>
#include <linux/mm.h>

/*
 * Exported VM tunables we adjust during game mode:
 *   vm_dirty_ratio              — max dirty page % before throttling writers
 *   dirty_background_ratio      — % at which background writeback starts
 *   dirty_writeback_interval    — periodic flusher wakeup interval (cs)
 *   dirty_expire_interval       — max age of dirty pages before writeback (cs)
 *   min_free_kbytes             — reserved free page pool size
 *   sysctl_vfs_cache_pressure   — how aggressively inode/dentry cache is reclaimed
 */
extern int vm_dirty_ratio;
extern int dirty_background_ratio;
extern unsigned int dirty_writeback_interval;
extern unsigned int dirty_expire_interval;
extern int min_free_kbytes;

/*
 * Debug logging gated behind CONFIG_ZENITH_DEBUG_MSG.
 * Production (community) builds compile these out entirely.
 */
#ifdef CONFIG_ZENITH_DEBUG_MSG
static bool debug_log;
module_param(debug_log, bool, 0644);
MODULE_PARM_DESC(debug_log, "Enable verbose dmesg logging (default: false)");

#define gpu_debug(fmt, ...)	pr_debug("zenith_gpu_switch: " fmt, ##__VA_ARGS__)
#else
static const bool debug_log = false;
#define gpu_debug(fmt, ...)	no_printk(KERN_DEBUG "zenith_gpu_switch: " fmt, ##__VA_ARGS__)
#endif

/***** Tunables *****/

static char gpu_devfreq_name[DEVFREQ_NAME_LEN] = "";
module_param_string(gpu_devfreq_name, gpu_devfreq_name,
		    sizeof(gpu_devfreq_name), 0644);
MODULE_PARM_DESC(gpu_devfreq_name,
		 "GPU devfreq device name (empty = auto-detect)");

static unsigned int gpu_check_ms = 1000;
module_param(gpu_check_ms, uint, 0644);
MODULE_PARM_DESC(gpu_check_ms,
		 "Poll interval in ms (default: 1000)");

static char gpu_game_governor[DEVFREQ_NAME_LEN] = "performance";
module_param_string(gpu_game_governor, gpu_game_governor,
		    sizeof(gpu_game_governor), 0644);
MODULE_PARM_DESC(gpu_game_governor,
		 "Governor while game mode is active (default: performance)");

static char gpu_active_governor[DEVFREQ_NAME_LEN] = "simple_ondemand";
module_param_string(gpu_active_governor, gpu_active_governor,
		    sizeof(gpu_active_governor), 0644);
MODULE_PARM_DESC(gpu_active_governor,
		 "Governor while GPU active, not gaming (default: simple_ondemand)");

static char gpu_idle_governor[DEVFREQ_NAME_LEN] = "simple_ondemand";
module_param_string(gpu_idle_governor, gpu_idle_governor,
		    sizeof(gpu_idle_governor), 0644);
MODULE_PARM_DESC(gpu_idle_governor,
		 "Governor after idle timeout (default: simple_ondemand)");

static unsigned int gpu_idle_ms = 5000;
module_param(gpu_idle_ms, uint, 0644);
MODULE_PARM_DESC(gpu_idle_ms,
		 "Idle timeout in ms before idle governor (default: 5000, 0=skip)");

/***** Game-mode memory tuning save state *****/

static int saved_dirty_ratio;
static int saved_dirty_bg_ratio;
static int saved_vfs_cache_pressure;
static unsigned int saved_dirty_writeback_interval;
static unsigned int saved_dirty_expire_interval;
static int saved_min_free_kbytes;
static bool game_tuning_active;

/*
 * During game mode, tighten dirty ratios and reduce VFS cache pressure
 * so that:
 *   - Writeback starts earlier and stays gentler  (smaller bursts)
 *   - File-backed pages (game assets) are evicted less aggressively
 *   - Dirty pages are flushed sooner to avoid a post-game fsync storm
 *   - More free memory is reserved for atomic allocations
 *
 * The normal defaults are dirty_ratio=20, dirty_bg=10, dirty_wb=1500cs,
 * dirty_expire=3000cs, min_free=~8-16MB, vfs_cache=100.
 * User-modified values are saved on game-enter and restored on game-exit.
 */
#define GAME_DIRTY_RATIO               10
#define GAME_DIRTY_BG_RATIO             5
#define GAME_VFS_CACHE_PRESSURE        50
#define GAME_DIRTY_WB_INTERVAL       300  /* 3 seconds in cs */
#define GAME_DIRTY_EXPIRE_INTERVAL   500  /* 5 seconds in cs */

/***** Internal state *****/

static struct delayed_work gpu_governor_work;
static unsigned long gpu_last_game_active_jiffies;
static bool gpu_prev_game_active;

/*
 * Resolve the GPU devfreq device.  If gpu_devfreq_name is set (non-empty),
 * use it as an exact match.  Otherwise auto-detect by scanning all devfreq
 * devices for GPU keywords (mali, gpu, kgsl, adreno, panfrost, ...).
 */
static struct devfreq *gpu_resolve_devfreq(void)
{
	if (gpu_devfreq_name[0])
		return devfreq_get_devfreq_by_name(gpu_devfreq_name);

	return devfreq_find_gpu_devfreq();
}

static void gpu_governor_worker(struct work_struct *work)
{
	struct devfreq *df;
	const char *target;
	bool game_active;
	unsigned long idle_jiffies;
	unsigned long total_ram_kb;

	df = gpu_resolve_devfreq();
	if (IS_ERR(df)) {
		/* GPU not probed yet -- retry later */
		goto resched;
	}

	game_active = zenith_is_game_mode_active();

	/*
	 * Proactively wake kcompactd when game mode first activates.
	 * This pre-compacts memory before game assets need high-order
	 * (DMA/GPU) allocations, reducing launch-time stalls.
	 */
	if (game_active && !gpu_prev_game_active)
		wakeup_all_kcompactd();

	/*
	 * Game-mode memory tuning: tighten dirty ratios and reduce VFS cache
	 * pressure so writeback stays gentle and game assets remain cached.
	 *
	 * Safety: the min_free_kbytes boost is capped to never exceed 5% of
	 * total RAM.  On a 4 GB device that's ~200 MB -- the 5 MB bump is
	 * well within range, but the cap prevents pathological over-reservation
	 * on very low-RAM or misconfigured systems.
	 */
	if (game_active && !game_tuning_active) {
		/* Save current values */
		saved_dirty_ratio = vm_dirty_ratio;
		saved_dirty_bg_ratio = dirty_background_ratio;
		saved_vfs_cache_pressure = sysctl_vfs_cache_pressure;
		saved_dirty_writeback_interval = dirty_writeback_interval;
		saved_dirty_expire_interval = dirty_expire_interval;
		saved_min_free_kbytes = min_free_kbytes;

		/* Apply game-mode values */
		vm_dirty_ratio = GAME_DIRTY_RATIO;
		dirty_background_ratio = GAME_DIRTY_BG_RATIO;
		sysctl_vfs_cache_pressure = GAME_VFS_CACHE_PRESSURE;
		dirty_writeback_interval = GAME_DIRTY_WB_INTERVAL;
		dirty_expire_interval = GAME_DIRTY_EXPIRE_INTERVAL;

		total_ram_kb = (unsigned long)totalram_pages() * (PAGE_SIZE / 1024);
		min_free_kbytes = min(min_free_kbytes + 5120,
				       (int)(total_ram_kb * 5 / 100));

		game_tuning_active = true;

		gpu_debug("game mode mem tuning ON "
			 "(dirty_ratio=%d, dirty_bg=%d, vfs_cache=%d, "
			 "wb=%ucs, expire=%ucs, min_free=%d)\n",
			 vm_dirty_ratio, dirty_background_ratio,
			 sysctl_vfs_cache_pressure,
			 dirty_writeback_interval, dirty_expire_interval,
			 min_free_kbytes);
	} else if (!game_active && game_tuning_active) {
		/* Restore saved values */
		vm_dirty_ratio = saved_dirty_ratio;
		dirty_background_ratio = saved_dirty_bg_ratio;
		sysctl_vfs_cache_pressure = saved_vfs_cache_pressure;
		dirty_writeback_interval = saved_dirty_writeback_interval;
		dirty_expire_interval = saved_dirty_expire_interval;
		min_free_kbytes = saved_min_free_kbytes;

		game_tuning_active = false;

		gpu_debug("game mode mem tuning OFF "
			 "(dirty_ratio=%d, dirty_bg=%d, vfs_cache=%d, "
			 "wb=%ucs, expire=%ucs, min_free=%d)\n",
			 vm_dirty_ratio, dirty_background_ratio,
			 sysctl_vfs_cache_pressure,
			 dirty_writeback_interval, dirty_expire_interval,
			 min_free_kbytes);
	}

	gpu_prev_game_active = game_active;

	if (game_active) {
		target = gpu_game_governor;
		gpu_last_game_active_jiffies = jiffies;
	} else {
		/*
		 * GPU activity detection: if the GPU's current frequency is
		 * above the minimum OPP, the GPU is doing real work even
		 * outside of game mode (scrolling, camera, UI rendering).
		 * Reset the idle timer so the governor stays on
		 * simple_ondemand instead of falling to the idle governor.
		 *
		 * This works correctly with simple_ondemand as the idle
		 * governor because ondemand scales frequency dynamically —
		 * it won't keep the GPU stuck at minimum when there's work.
		 */
		if (df->previous_freq > df->scaling_min_freq) {
			gpu_last_game_active_jiffies = jiffies;
			gpu_debug("%s: GPU busy (freq=%lu > min=%lu), reset idle\n",
				  dev_name(&df->dev),
				  df->previous_freq, df->scaling_min_freq);
		}

		idle_jiffies = jiffies - gpu_last_game_active_jiffies;
		if (gpu_idle_ms > 0 &&
		    jiffies_to_msecs(idle_jiffies) >= gpu_idle_ms)
			target = gpu_idle_governor;
		else
			target = gpu_active_governor;
	}

	/* Only switch if the governor actually changed */
	if (strcmp(df->governor_name, target)) {
		pr_info("zenith_gpu_switch: %s: %s -> %s%s\n",
			dev_name(&df->dev), df->governor_name, target,
			game_active ? " (game on)" : "");
		devfreq_set_governor(df, target);
	} else {
		gpu_debug("%s: already %s (game=%d)\n",
			  dev_name(&df->dev), target, game_active);
	}

resched:
	queue_delayed_work(system_unbound_wq, &gpu_governor_work,
			   msecs_to_jiffies(gpu_check_ms));
}

static int __init zenith_gpu_switch_init(void)
{
	gpu_last_game_active_jiffies = jiffies;
	INIT_DELAYED_WORK(&gpu_governor_work, gpu_governor_worker);
	queue_delayed_work(system_unbound_wq, &gpu_governor_work,
			   msecs_to_jiffies(gpu_check_ms));

	pr_info("zenith_gpu_switch: watching '%s' every %u ms\n",
		gpu_devfreq_name, gpu_check_ms);
	return 0;
}

static void __exit zenith_gpu_switch_exit(void)
{
	cancel_delayed_work_sync(&gpu_governor_work);
	pr_info("zenith_gpu_switch: stopped\n");
}

module_init(zenith_gpu_switch_init);
module_exit(zenith_gpu_switch_exit);

MODULE_DESCRIPTION("Zenith GPU devfreq governor auto-switcher");
MODULE_AUTHOR("Codebuff");
MODULE_LICENSE("GPL v2");
