// SPDX-License-Identifier: GPL-2.0
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/mm.h>
#include <linux/device.h>
#include <linux/version.h>

#define DEVICE_NAME     "vtophys"
#define CLASS_NAME      "vtophys"
#define VTIOCTL_GET_PHYS _IOR('p', 1, unsigned long)

/* ───────── ioctl core ───────── */
static long vtophys_ioctl(struct file *f, unsigned int cmd, unsigned long arg)
{
    unsigned long vaddr, phys;
    struct page *pg;
    int ret;

    if (cmd != VTIOCTL_GET_PHYS) return -EINVAL;

    if (copy_from_user(&vaddr, (void __user *)arg, sizeof(vaddr)))
        return -EFAULT;

    /* Pin the user page */
    mmap_read_lock(current->mm);
    ret = get_user_pages(vaddr, 1, FOLL_GET, &pg);
    mmap_read_unlock(current->mm);

    if (ret != 1) return -EFAULT;

    /* Convert page to physical address */
    phys = (page_to_pfn(pg) << PAGE_SHIFT) | (vaddr & ~PAGE_MASK);
    put_page(pg);

    if (copy_to_user((void __user *)arg, &phys, sizeof(phys)))
        return -EFAULT;

    return 0;
}
/* ───────── file ops / boilerplate ───────── */
static const struct file_operations vtophys_fops = {
    .owner          = THIS_MODULE,
    .unlocked_ioctl = vtophys_ioctl,
    .compat_ioctl   = vtophys_ioctl,
};

static int major;
static struct class *vtophys_class;

static int __init vtophys_init(void)
{
    major = register_chrdev(0, DEVICE_NAME, &vtophys_fops);
    if (major < 0) return major;

    vtophys_class = class_create(CLASS_NAME);
    if (IS_ERR(vtophys_class)) {
        unregister_chrdev(major, DEVICE_NAME);
        return PTR_ERR(vtophys_class);
    }
    device_create(vtophys_class, NULL, MKDEV(major, 0), NULL, DEVICE_NAME);
    pr_info("vtophys: loaded (major %d)\n", major);
    return 0;
}

static void __exit vtophys_exit(void)
{
    device_destroy(vtophys_class, MKDEV(major, 0));
    class_destroy(vtophys_class);
    unregister_chrdev(major, DEVICE_NAME);
    pr_info("vtophys: unloaded\n");
}

module_init(vtophys_init);
module_exit(vtophys_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("educational example");
MODULE_DESCRIPTION("User VA → phys address demo");
