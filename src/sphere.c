#include <stdio.h>  // printf, FILE
#include <stdlib.h> // rand, malloc, free
#include <math.h>   //
#include <time.h>

#define M_PI 3.14159265358979323846

int main(int argc, char *argv)
{
    int n = 100000;

    float *points_h = (float *)malloc(n * 3 * sizeof(float));

    srand(time(NULL));

    for (int i = 0; i < n; i++)
    {
        float theta = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
        float phi = ((float)rand() / RAND_MAX) * M_PI;

        float r = 1.0f;

        points_h[3 * i + 0] = r * sinf(phi) * cosf(theta);
        points_h[3 * i + 1] = r * sinf(phi) * sinf(theta);
        points_h[3 * i + 2] = r * cosf(phi);
    }

    // ----------------------------
    // write PCD file for PCL
    // ----------------------------
    FILE *fp = fopen("../sphere.pcd", "w");

    fprintf(fp,
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z\n"
            "SIZE 4 4 4\n"
            "TYPE F F F\n"
            "COUNT 1 1 1\n"
            "WIDTH %d\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            "POINTS %d\n"
            "DATA ascii\n",
            n, n);

    for (int i = 0; i < n; i++)
    {
        fprintf(fp, "%f %f %f\n",
                points_h[3 * i + 0],
                points_h[3 * i + 1],
                points_h[3 * i + 2]);
    }

    fclose(fp);

    printf("Saved sphere.pcd with %d points\n", n);

    free(points_h);

    return 0;
}