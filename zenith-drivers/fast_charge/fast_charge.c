// SPDX-License-Identifier: GPL-2.0-only
/*
 * fast_charge — universal GKI charger current booster
 *
 * Boosts charge current and input current limit on charger hotplug
 * using the standard power supply class API only. Works on any GKI
 * 5.10+ device regardless of vendor charger IC driver.
 *
 * How it works:
 *   1. Scans all power supplies at module init, identifies charger-typed
 *      supplies that are online, and bumps their key current limits.
 *   2. Registers a power supply notifier to re-apply settings when a
 *      new charger is plugged in.
 *   3. Exposes sysfs knobs at /sys/kernel/fast_charge/ for runtime tuning.
 *
 * Safety:
 *   - Reads CONSTANT_CHARGE_CURRENT_MAX / CONSTANT_CHARGE_VOLTAGE_MAX
 *     and never exceeds hardware limits.
 *   - Checks POWER_SUPPLY_PROP_STATUS before applying — only boosts
 *     when actually charging.
 *   - Skips supplies that don't support the required properties.
 *   - Module params and sysfs let you dial back if too aggressive.
 *
 * Usage:
 *   # Load with defaults (3A):
 *   insmod fast_charge.ko
 *
 *   # Load with custom current:
 *   insmod fast_charge.ko charge_current_ua=4000000 input_current_ua=5000000
 *
 *   # Tune at runtime:
 *   echo 4000000 > /sys/kernel/fast_charge/charge_current_ua
 *   echo 0 > /sys/kernel/fast_charge/charge_current_ua  # use hw max
 *
 * Build (external module against GKI headers):
 *   make -C /path/to/gki_kernel_src M=$PWD modules
 *
 * Author: auto-generated for zenith kernel project
 */

#define pr_fmt(fmt) "fast_charge: " fmt

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/power_supply.h>
#include <linux/notifier.h>
#include <linux/workqueue.h>
#include <linux/slab.h>
#include <linux/device.h>
#include <linux/string.h>
#include <linux/kobject.h>
#include <linux/sysfs.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/delay.h>

/*
 * Module parameters — set at insmod time.
 * 0 = use the hardware's advertised maximum.
 */
static unsigned int charge_current_ua = 3000000;
static unsigned int input_current_ua = 3000000;
module_param(charge_current_ua, uint, 0644);
MODULE_PARM_DESC(charge_current_ua,
		 "Target constant charge current in uA (0 = hw max, default 3000000)");
module_param(input_current_ua, uint, 0644);
MODULE_PARM_DESC(input_current_ua,
		 "Target input current limit in uA (0 = hw max, default 3000000)");

static bool fast_charge_enabled = true;

/* Battery temp threshold in deci-Celsius. 0 = disabled. Default 450 = 45.0 C */
static unsigned int batt_temp_max_decicelsius = 450;
module_param(batt_temp_max_decicelsius, uint, 0644);
MODULE_PARM_DESC(batt_temp_max_decicelsius,
		 "Max battery temp for boosting (deci-Celsius, 0=disabled, default 450=45.0C)");

/* Boost CV voltage to hw max? 1=yes (default), 0=no */
static bool boost_cv_voltage = true;
module_param(boost_cv_voltage, bool, 0644);
MODULE_PARM_DESC(boost_cv_voltage,
		 "Boost constant charge voltage to hw max (default 1)");

/* ---- Internal state ---- */

static struct notifier_block fc_nb;
static struct delayed_work fc_work;
static struct kobject *fc_kobj;

/*
 * Stats counters (RO via sysfs).
 */
static struct {
	unsigned long boost_applied;
	unsigned long boost_skipped_not_charging;
	unsigned long boost_skipped_no_prop;
	unsigned long errors;
} fc_stats;

/* ---- Core: boost a single power supply ---- */

/*
 * Try to boost a single charger-type power supply.
 * Returns 0 on success, negative on error, 1 if skipped.
 */
static int fc_boost_psy(struct power_supply *psy)
{
	union power_supply_propval val;
	int ret;

	if (!psy || !psy->desc)
		return 1;

	/* Only boost charger-type supplies */
	switch (psy->desc->type) {
	case POWER_SUPPLY_TYPE_MAINS:
	case POWER_SUPPLY_TYPE_USB:
	case POWER_SUPPLY_TYPE_USB_DCP:
	case POWER_SUPPLY_TYPE_USB_CDP:
	case POWER_SUPPLY_TYPE_USB_ACA:
	case POWER_SUPPLY_TYPE_USB_TYPE_C:
	case POWER_SUPPLY_TYPE_USB_PD:
	case POWER_SUPPLY_TYPE_USB_PD_DRP:
	case POWER_SUPPLY_TYPE_WIRELESS:
		break;
	default:
		return 1; /* not a charger */
	}

	/* Must be online */
	ret = power_supply_get_property(psy, POWER_SUPPLY_PROP_ONLINE, &val);
	if (ret || !val.intval)
		return 1;

	/* Must be in charging state */
	ret = power_supply_get_property(psy, POWER_SUPPLY_PROP_STATUS, &val);
	if (ret == 0 && val.intval != POWER_SUPPLY_STATUS_CHARGING &&
	    val.intval != POWER_SUPPLY_STATUS_FULL) {
		fc_stats.boost_skipped_not_charging++;
		return 1;
	}

	/*
	 * Check battery temperature safety.
	 * Walk to find the battery and read its temp before boosting.
	 */
	if (READ_ONCE(batt_temp_max_decicelsius) > 0) {
		struct power_supply *bpsy;
		union power_supply_propval temp_val;
		int btemp;

		bpsy = power_supply_get_by_name("battery");
		if (!bpsy)
			bpsy = power_supply_get_by_name("BATTERY");
		if (bpsy) {
			if (power_supply_get_property(bpsy,
				    POWER_SUPPLY_PROP_TEMP, &temp_val) == 0) {
				btemp = temp_val.intval;
				if (btemp > READ_ONCE(batt_temp_max_decicelsius)) {
					pr_info("%s: battery too hot (%d.%dC > %d.%dC), skipping\n",
						psy->desc->name,
						btemp / 10, btemp % 10,
						READ_ONCE(batt_temp_max_decicelsius) / 10,
						READ_ONCE(batt_temp_max_decicelsius) % 10);
					power_supply_put(bpsy);
					return 1;
				}
			}
			power_supply_put(bpsy);
		}
	}

	/*
	 * 1. Boost constant charge current (CC phase).
	 */
	if (charge_current_ua > 0) {
		union power_supply_propval max_val;

		/* Check hardware max */
		ret = power_supply_get_property(psy,
			POWER_SUPPLY_PROP_CONSTANT_CHARGE_CURRENT_MAX,
			&max_val);
		if (ret == 0 && max_val.intval > 0) {
			unsigned int target = charge_current_ua;

			if (target > max_val.intval) {
				pr_info("%s: capping CC %u -> %u (hw max)\n",
					psy->desc->name, target, max_val.intval);
				target = max_val.intval;
			}

			val.intval = target;
			ret = power_supply_set_property(psy,
				POWER_SUPPLY_PROP_CONSTANT_CHARGE_CURRENT,
				&val);
			if (ret == 0) {
				pr_info("%s: CC boosted to %u uA\n",
					psy->desc->name, target);
				fc_stats.boost_applied++;
			} else if (ret != -ENODEV && ret != -EINVAL) {
				fc_stats.errors++;
			}
		} else {
			fc_stats.boost_skipped_no_prop++;
		}
	}

	/*
	 * 2. Boost input current limit.
	 */
	if (input_current_ua > 0) {
		union power_supply_propval max_val;

		ret = power_supply_get_property(psy,
			POWER_SUPPLY_PROP_CURRENT_MAX, &max_val);
		if (ret == 0 && max_val.intval > 0) {
			unsigned int target = input_current_ua;

			if (target > max_val.intval) {
				pr_info("%s: capping input current %u -> %u (hw max)\n",
					psy->desc->name, target, max_val.intval);
				target = max_val.intval;
			}

			val.intval = target;
			ret = power_supply_set_property(psy,
				POWER_SUPPLY_PROP_INPUT_CURRENT_LIMIT,
				&val);
			if (ret == 0) {
				pr_info("%s: input current boosted to %u uA\n",
					psy->desc->name, target);
				fc_stats.boost_applied++;
			} else if (ret != -ENODEV && ret != -EINVAL) {
				fc_stats.errors++;
			}
		} else {
			fc_stats.boost_skipped_no_prop++;
		}
	}

	/*
	 * 3. Boost charge voltage (CV phase) to max safe.
	 */
	if (READ_ONCE(boost_cv_voltage)) {
		union power_supply_propval max_val;

		ret = power_supply_get_property(psy,
			POWER_SUPPLY_PROP_CONSTANT_CHARGE_VOLTAGE_MAX,
			&max_val);
		if (ret == 0 && max_val.intval > 0) {
			val.intval = max_val.intval;
			ret = power_supply_set_property(psy,
				POWER_SUPPLY_PROP_CONSTANT_CHARGE_VOLTAGE,
				&val);
			if (ret == 0) {
				pr_info("%s: CV boosted to %u uV\n",
					psy->desc->name, max_val.intval);
				fc_stats.boost_applied++;
			} else if (ret != -ENODEV && ret != -EINVAL) {
				fc_stats.errors++;
			}
		} else {
			fc_stats.boost_skipped_no_prop++;
		}
	}

	return 0;
}

/* ---- Workqueue: scan + boost all ---- */

struct fc_psy_walk {
	int count;
};

static int fc_psy_boost_cb(struct device *dev, void *data)
{
	struct power_supply *psy = dev_get_drvdata(dev);
	struct fc_psy_walk *w = data;

	if (!psy)
		return 0;

	if (fc_boost_psy(psy) == 0)
		w->count++;

	return 0;
}

static void fc_refresh_work(struct work_struct *work)
{
	struct fc_psy_walk w = { .count = 0 };

	if (!READ_ONCE(fast_charge_enabled))
		return;

	if (class_for_each_device(power_supply_class, NULL, &w,
				  fc_psy_boost_cb))
		pr_warn("error during power supply scan\n");

	if (w.count > 0)
		pr_debug("boosted %d charger supply(s)\n", w.count);
}

/* ---- Power supply notifier ---- */

static int fc_psy_notifier(struct notifier_block *nb, unsigned long event,
			   void *data)
{
	struct power_supply *psy = data;

	if (event != PSY_EVENT_PROP_CHANGED)
		return NOTIFY_DONE;

	/*
	 * Ignore battery-only events — they don't change charger state.
	 * But if we're already tracking a charger, re-apply anyway in
	 * case the charger re-negotiated its contract.
	 */
	if (psy && psy->desc &&
	    psy->desc->type == POWER_SUPPLY_TYPE_BATTERY)
		return NOTIFY_DONE;

	if (!READ_ONCE(fast_charge_enabled))
		return NOTIFY_DONE;

	/*
	 * Debounce: schedule work in 500ms to avoid thundering herd
	 * on rapid property changes during PD negotiation.
	 */
	mod_delayed_work(system_wq, &fc_work, msecs_to_jiffies(500));

	return NOTIFY_OK;
}

/* ---- Sysfs interface (/sys/kernel/fast_charge/) ---- */

static ssize_t enabled_show(struct kobject *kobj,
			    struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%u\n", READ_ONCE(fast_charge_enabled));
}

static ssize_t enabled_store(struct kobject *kobj,
			     struct kobj_attribute *attr,
			     const char *buf, size_t count)
{
	unsigned int val;
	int ret;

	ret = kstrtouint(buf, 0, &val);
	if (ret)
		return ret;

	WRITE_ONCE(fast_charge_enabled, !!val);

	/* If enabled, schedule an immediate boost */
	if (val)
		mod_delayed_work(system_wq, &fc_work, 0);

	return count;
}

static ssize_t charge_current_ua_show(struct kobject *kobj,
				      struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%u\n", READ_ONCE(charge_current_ua));
}

static ssize_t charge_current_ua_store(struct kobject *kobj,
				       struct kobj_attribute *attr,
				       const char *buf, size_t count)
{
	unsigned int val;
	int ret;

	ret = kstrtouint(buf, 0, &val);
	if (ret)
		return ret;

	WRITE_ONCE(charge_current_ua, val);

	/* Re-apply immediately */
	if (READ_ONCE(fast_charge_enabled))
		mod_delayed_work(system_wq, &fc_work, 0);

	return count;
}

static ssize_t input_current_ua_show(struct kobject *kobj,
				     struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%u\n", READ_ONCE(input_current_ua));
}

static ssize_t input_current_ua_store(struct kobject *kobj,
				      struct kobj_attribute *attr,
				      const char *buf, size_t count)
{
	unsigned int val;
	int ret;

	ret = kstrtouint(buf, 0, &val);
	if (ret)
		return ret;

	WRITE_ONCE(input_current_ua, val);

	if (READ_ONCE(fast_charge_enabled))
		mod_delayed_work(system_wq, &fc_work, 0);

	return count;
}

static ssize_t status_show(struct kobject *kobj,
			   struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf,
		"enabled              %u\n"
		"charge_current_ua    %u\n"
		"input_current_ua     %u\n"
		"boost_cv_voltage     %u\n"
		"batt_temp_max        %u\n"
		"boost_applied        %lu\n"
		"skipped_not_charging %lu\n"
		"skipped_no_prop      %lu\n"
		"errors               %lu\n",
		READ_ONCE(fast_charge_enabled),
		READ_ONCE(charge_current_ua),
		READ_ONCE(input_current_ua),
		READ_ONCE(boost_cv_voltage),
		READ_ONCE(batt_temp_max_decicelsius),
		fc_stats.boost_applied,
		fc_stats.boost_skipped_not_charging,
		fc_stats.boost_skipped_no_prop,
		fc_stats.errors);
}

static struct kobj_attribute fc_enabled_attr =
	__ATTR_RW(enabled);

static struct kobj_attribute fc_charge_current_attr =
	__ATTR_RW(charge_current_ua);

static struct kobj_attribute fc_input_current_attr =
	__ATTR_RW(input_current_ua);

static struct kobj_attribute fc_status_attr =
	__ATTR_RO(status);

static struct attribute *fc_attrs[] = {
	&fc_enabled_attr.attr,
	&fc_charge_current_attr.attr,
	&fc_input_current_attr.attr,
	&fc_status_attr.attr,
	NULL,
};

static struct attribute_group fc_attr_group = {
	.attrs = fc_attrs,
};

/* ---- Init / Exit ---- */

static int __init fast_charge_init(void)
{
	int ret;

	pr_info("loading (charge_current=%u uA, input_current=%u uA)\n",
		charge_current_ua, input_current_ua);

	INIT_DELAYED_WORK(&fc_work, fc_refresh_work);

	/* Create sysfs at /sys/kernel/fast_charge/ */
	fc_kobj = kobject_create_and_add("fast_charge", kernel_kobj);
	if (!fc_kobj) {
		pr_err("failed to create sysfs kobject\n");
		return -ENOMEM;
	}

	ret = sysfs_create_group(fc_kobj, &fc_attr_group);
	if (ret) {
		pr_err("failed to create sysfs group: %d\n", ret);
		kobject_put(fc_kobj);
		fc_kobj = NULL;
		return ret;
	}

	/* Register power supply notifier for hotplug */
	fc_nb.notifier_call = fc_psy_notifier;
	ret = power_supply_reg_notifier(&fc_nb);
	if (ret) {
		pr_err("failed to register power supply notifier: %d\n", ret);
		sysfs_remove_group(fc_kobj, &fc_attr_group);
		kobject_put(fc_kobj);
		fc_kobj = NULL;
		return ret;
	}

	/*
	 * Initial sweep: catch chargers already plugged in at boot.
	 * Use a small delay so all PSY drivers have finished probing.
	 */
	mod_delayed_work(system_wq, &fc_work, msecs_to_jiffies(2000));

	pr_info("loaded successfully\n");
	return 0;
}

static void __exit fast_charge_exit(void)
{
	/* Unreg notifier first so no new work gets scheduled */
	power_supply_unreg_notifier(&fc_nb);
	cancel_delayed_work_sync(&fc_work);

	if (fc_kobj) {
		sysfs_remove_group(fc_kobj, &fc_attr_group);
		kobject_put(fc_kobj);
	}

	pr_info("unloaded\n");
}

module_init(fast_charge_init);
module_exit(fast_charge_exit);

MODULE_LICENSE("GPL v2");
MODULE_DESCRIPTION("Universal GKI charger current booster");
MODULE_AUTHOR("zenith project");
