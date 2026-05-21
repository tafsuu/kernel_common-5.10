// SPDX-License-Identifier: GPL-2.0
/*
 * thermal_charger_guard (霞ガード) -- auto-suppress Kasumi and Iyashi
 * while an external charger is online.  -- XTENSEI
 *
 * Background:
 *   Some Transsion userspace battery managers read post-Kasumi
 *   dampened thermal_zone temperatures and refuse to start charging
 *   until that reading falls inside an expected band.  Because Kasumi
 *   lags the real temperature on purpose, the band check spins and
 *   the device throws "won't charge until normal temps" warnings on a
 *   battery that is otherwise healthy.  Iyashi can produce a similar
 *   surprise for any userspace that watches cooling-device cur_state.
 *
 *   Pragmatic fix: whenever any MAINS / USB-family / WIRELESS power
 *   supply reports ONLINE, suppress both subsystems for the duration
 *   of the charging session.  Restore on unplug.
 *
 * Layering:
 *   This module does NOT touch /sys/kernel/{kasumi,iyashi}/enabled.
 *   It carries an orthogonal "charger_suppressed" flag on each side.
 *   The effective active state is (enabled && !charger_suppressed).
 *   Userspace can still toggle `enabled` at will; the suppression
 *   flag is visible read-only at /sys/kernel/{kasumi,iyashi}/
 *   charger_suppressed for observability.
 *
 * Debouncing:
 *   PSY_EVENT_PROP_CHANGED fires often (battery SOC updates, ICL
 *   re-negotiation, temp polling) so the notifier itself just kicks
 *   a delayed_work with a HZ/4 debounce.  The worker then walks
 *   every registered power_supply once and OR-reduces ONLINE across
 *   charger-typed psys to derive the aggregate state.
 *
 * Boot ordering:
 *   We register at late_initcall, the same level as kasumi/iyashi
 *   sysfs init.  The setters are no-ops while the target's static
 *   storage is still all-zero, so an early notifier callback before
 *   the target's init runs is harmless -- it just writes 0 to a 0.
 *   An initial refresh is kicked from init() to pick up devices
 *   already plugged in when the kernel finishes booting.
 */

#define pr_fmt(fmt) "thermal_charger_guard: " fmt

#include <linux/device.h>
#include <linux/init.h>
#include <linux/jiffies.h>
#include <linux/notifier.h>
#include <linux/power_supply.h>
#include <linux/printk.h>
#include <linux/spinlock.h>
#include <linux/workqueue.h>

#include "iyashi.h"
#include "kasumi.h"

#define THERMAL_CHARGER_GUARD_DEBOUNCE_MS 250

/*
 * Suppression tier, by charger type:
 *   0 = offline (no suppression)
 *   1 = standard charger (suppress Kasumi only -- fixes Transsion
 *       battery-manager band check on USB/DCP/MAINS)
 *   2 = fast / wireless charger (suppress both Kasumi and Iyashi)
 */
static unsigned int charger_tier __read_mostly;
static DEFINE_SPINLOCK(charger_state_lock);

static struct delayed_work charger_refresh_work;

/*
 * Map a power_supply type to its suppression tier.
 * Non-charger types return 0 (ignored).
 */
static unsigned int psy_charger_tier(struct power_supply *psy)
{
	if (!psy || !psy->desc)
		return 0;

	switch (psy->desc->type) {
	/* Tier 2: fast charging / wireless */
	case POWER_SUPPLY_TYPE_USB_PD:
	case POWER_SUPPLY_TYPE_USB_PD_DRP:
	case POWER_SUPPLY_TYPE_WIRELESS:
		return 2;

	/* Tier 1: standard chargers (MAINS, USB, legacy) */
	case POWER_SUPPLY_TYPE_MAINS:
	case POWER_SUPPLY_TYPE_USB:
	case POWER_SUPPLY_TYPE_USB_DCP:
	case POWER_SUPPLY_TYPE_USB_CDP:
	case POWER_SUPPLY_TYPE_USB_ACA:
	case POWER_SUPPLY_TYPE_USB_TYPE_C:
	case POWER_SUPPLY_TYPE_APPLE_BRICK_ID:
		return 1;

	default:
		return 0;
	}
}

struct charger_walk {
	unsigned int max_tier;
};

static int charger_psy_check(struct device *dev, void *data)
{
	struct charger_walk *w = data;
	struct power_supply *psy = dev_get_drvdata(dev);
	union power_supply_propval val;
	unsigned int tier;

	tier = psy_charger_tier(psy);
	if (!tier)
		return 0;

	if (power_supply_get_property(psy, POWER_SUPPLY_PROP_ONLINE, &val))
		return 0;

	if (val.intval && tier > w->max_tier)
		w->max_tier = tier;

	return 0;
}

static void charger_refresh(struct work_struct *work)
{
	struct charger_walk w = { .max_tier = 0 };
	unsigned int prev;

	if (class_for_each_device(power_supply_class, NULL, &w,
				  charger_psy_check))
		return;

	spin_lock(&charger_state_lock);
	prev = charger_tier;
	charger_tier = w.max_tier;
	spin_unlock(&charger_state_lock);

	if (prev == w.max_tier)
		return;

	/*
	 * Tier 1: suppress Kasumi only (fixes Transsion battery-manager
	 *          band check on USB/DCP/MAINS).
	 * Tier 2: suppress both Kasumi and Iyashi.
	 * Tier 0: restore both.
	 */
	kasumi_set_charger_suppressed(w.max_tier >= 1);
	iyashi_set_charger_suppressed(w.max_tier >= 2);

	pr_info("charger tier %u -> kasumi %s, iyashi %s\n",
		w.max_tier,
		w.max_tier >= 1 ? "suppressed" : "restored",
		w.max_tier >= 2 ? "suppressed" : "restored");
}

static int charger_psy_notify(struct notifier_block *nb, unsigned long event,
			      void *data)
{
	struct power_supply *psy = data;

	if (event != PSY_EVENT_PROP_CHANGED)
		return NOTIFY_DONE;

	/*
	 * Battery-only psys never change our aggregate, but their SOC /
	 * temperature props churn the most.  Filter them out to keep
	 * this off the hot path entirely.  An unknown-type psy still
	 * triggers a refresh -- we'd rather re-walk once than miss a
	 * charger that mis-typed itself as UNKNOWN.
	 */
	if (psy && psy->desc &&
	    psy->desc->type == POWER_SUPPLY_TYPE_BATTERY)
		return NOTIFY_DONE;

	mod_delayed_work(system_wq, &charger_refresh_work,
			 msecs_to_jiffies(THERMAL_CHARGER_GUARD_DEBOUNCE_MS));
	return NOTIFY_OK;
}

static struct notifier_block charger_psy_nb = {
	.notifier_call = charger_psy_notify,
};

static int __init thermal_charger_guard_init(void)
{
	int ret;

	INIT_DELAYED_WORK(&charger_refresh_work, charger_refresh);

	ret = power_supply_reg_notifier(&charger_psy_nb);
	if (ret) {
		pr_err("power_supply_reg_notifier failed: %d\n", ret);
		return ret;
	}

	/*
	 * Initial sweep: catch the case where the device was plugged in
	 * before boot finished, so the suppression latches correctly
	 * without waiting for the next prop change.
	 */
	mod_delayed_work(system_wq, &charger_refresh_work, 0);

	pr_info("registered (debounce=%ums)\n",
		THERMAL_CHARGER_GUARD_DEBOUNCE_MS);
	return 0;
}
late_initcall(thermal_charger_guard_init);

/* ---- sysfs interface (/sys/kernel/thermal_charger_guard/) ---- */

static ssize_t tier_show(struct kobject *kobj,
			 struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%u\n", READ_ONCE(charger_tier));
}

static struct kobj_attribute thermal_charger_guard_tier_attr =
	__ATTR(tier, 0444, tier_show, NULL);

static struct attribute *thermal_charger_guard_attrs[] = {
	&thermal_charger_guard_tier_attr.attr,
	NULL,
};

static struct attribute_group thermal_charger_guard_attr_group = {
	.attrs = thermal_charger_guard_attrs,
};

static struct kobject *thermal_charger_guard_kobj;

static int __init thermal_charger_guard_sysfs_init(void)
{
	thermal_charger_guard_kobj = kobject_create_and_add(
		"thermal_charger_guard", kernel_kobj);
	if (!thermal_charger_guard_kobj)
		return -ENOMEM;

	return sysfs_create_group(thermal_charger_guard_kobj,
				  &thermal_charger_guard_attr_group);
}
/* later_initcall_sync to guarantee kasumi/iyashi sysfs init ran first */
late_initcall_sync(thermal_charger_guard_sysfs_init);
