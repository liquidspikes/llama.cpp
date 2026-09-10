#include "tbstripe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#define CONFIGFS_STREAM_PATH "/sys/kernel/config/thunderbolt/stream"
#define SYSFS_TB_DEVICES     "/sys/bus/thunderbolt/devices"

int tbs_sysfs_check_busy_poll(const char *xdomain, const char *stream) {
    char path[256];
    snprintf(path, sizeof(path), "%s/%s/%s/busy_poll", CONFIGFS_STREAM_PATH, xdomain, stream);
    
    FILE *f = fopen(path, "r");
    if (!f) {
        return -1;
    }
    int val = -1;
    if (fscanf(f, "%d", &val) != 1) {
        val = -1;
    }
    fclose(f);
    return val;
}

int tbs_sysfs_set_busy_poll(const char *xdomain, const char *stream, int enable) {
    char path[256];
    snprintf(path, sizeof(path), "%s/%s/%s/busy_poll", CONFIGFS_STREAM_PATH, xdomain, stream);
    
    FILE *f = fopen(path, "w");
    if (!f) {
        return -1;
    }
    int ret = (fprintf(f, "%d\n", enable ? 1 : 0) > 0) ? 0 : -1;
    fclose(f);
    return ret;
}

int tbs_sysfs_discover_xdomains(char *xdomain_a, size_t sz_a, char *xdomain_b, size_t sz_b) {
    DIR *dir = opendir(SYSFS_TB_DEVICES);
    if (!dir) {
        return -1;
    }

    struct dirent *entry;
    int found = 0;
    while ((entry = readdir(dir)) != NULL) {
        // Match xdomain patterns like "0-2.0" or "1-2.0"
        if (entry->d_name[0] == '.') continue;
        char *dot = strchr(entry->d_name, '.');
        char *dash = strchr(entry->d_name, '-');
        if (dash && dot && dot > dash) {
            if (found == 0 && xdomain_a && sz_a > 0) {
                strncpy(xdomain_a, entry->d_name, sz_a - 1);
                xdomain_a[sz_a - 1] = '\0';
                found++;
            } else if (found == 1 && xdomain_b && sz_b > 0) {
                strncpy(xdomain_b, entry->d_name, sz_b - 1);
                xdomain_b[sz_b - 1] = '\0';
                found++;
                break;
            }
        }
    }
    closedir(dir);
    return found;
}
