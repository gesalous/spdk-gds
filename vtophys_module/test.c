#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define VTIOCTL_GET_PHYS _IOR('p', 1, unsigned long)

int main() {
    int fd = open("/dev/vtophys", O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }

    unsigned long vaddr = (unsigned long)&vaddr;  /* any user VA */
    unsigned long phys  = vaddr;

    if (ioctl(fd, VTIOCTL_GET_PHYS, &phys) == -1) { perror("ioctl"); return 1; }

    printf("VA 0x%lx -> PA 0x%lx\n", vaddr, phys);
    close(fd);
    return 0;
}
