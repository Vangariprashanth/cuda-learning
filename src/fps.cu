#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define INF 1e9

// -----------------------------
// CUDA FPS kernel (distance update)
// -----------------------------
__global__ void kernel_fps(
    int centroid_idx,
    float *points_d,
    float *distance_d,
    int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n)
    {

        float cx = points_d[3 * centroid_idx + 0];
        float cy = points_d[3 * centroid_idx + 1];
        float cz = points_d[3 * centroid_idx + 2];

        float x = points_d[3 * index + 0];
        float y = points_d[3 * index + 1];
        float z = points_d[3 * index + 2];

        float dx = x - cx;
        float dy = y - cy;
        float dz = z - cz;

        float dist = dx * dx + dy * dy + dz * dz;

        if (dist < distance_d[index])
        {
            distance_d[index] = dist;
        }
    }
}

// -----------------------------
// MAIN
// -----------------------------
int main()
{

    // -----------------------------
    // Load PCD using PCL
    // -----------------------------
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile("sphere.pcd", *cloud) == -1)
    {
        printf("Could not load sphere.pcd\n");
        return -1;
    }

    int n = cloud->points.size();
    int sample_num = 1024;

    printf("Loaded points: %d\n", n);

    // -----------------------------
    // Flatten to array
    // -----------------------------
    float *points_h = (float *)malloc(n * 3 * sizeof(float));
    float *distance_h = (float *)malloc(n * sizeof(float));
    int *result_h = (int *)malloc(sample_num * sizeof(int));

    for (int i = 0; i < n; i++)
    {
        points_h[3 * i + 0] = cloud->points[i].x;
        points_h[3 * i + 1] = cloud->points[i].y;
        points_h[3 * i + 2] = cloud->points[i].z;

        distance_h[i] = INF;
    }

    // -----------------------------
    // GPU memory
    // -----------------------------
    float *points_d, *distance_d;

    cudaMalloc(&points_d, n * 3 * sizeof(float));
    cudaMalloc(&distance_d, n * sizeof(float));

    cudaMemcpy(points_d, points_h, n * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(distance_d, distance_h, n * sizeof(float), cudaMemcpyHostToDevice);

    // -----------------------------
    // FPS init
    // -----------------------------
    int centroid = 0;
    result_h[0] = centroid;

    printf("Starting FPS...\n");

    // -----------------------------
    // FPS LOOP
    // -----------------------------
    for (int i = 1; i < sample_num; i++)
    {

        kernel_fps<<<(n + 255) / 256, 256>>>(
            centroid,
            points_d,
            distance_d,
            n);

        cudaDeviceSynchronize();

        cudaMemcpy(distance_h, distance_d, n * sizeof(float), cudaMemcpyDeviceToHost);

        // argmax (CPU for simplicity)
        float max_dist = -1;
        int max_idx = 0;

        for (int j = 0; j < n; j++)
        {
            if (distance_h[j] > max_dist)
            {
                max_dist = distance_h[j];
                max_idx = j;
            }
        }

        centroid = max_idx;
        result_h[i] = centroid;
    }

    // -----------------------------
    // Build sampled cloud
    // -----------------------------
    pcl::PointCloud<pcl::PointXYZ>::Ptr sampled(new pcl::PointCloud<pcl::PointXYZ>);

    for (int i = 0; i < sample_num; i++)
    {
        sampled->points.push_back(cloud->points[result_h[i]]);
    }

    sampled->width = sample_num;
    sampled->height = 1;
    sampled->is_dense = true;

    // -----------------------------
    // Save result
    // -----------------------------
    pcl::io::savePCDFileASCII("sphere_fps.pcd", *sampled);

    printf("Saved sphere_fps.pcd with %d points\n", sample_num);

    // -----------------------------
    // cleanup
    // -----------------------------
    cudaFree(points_d);
    cudaFree(distance_d);

    free(points_h);
    free(distance_h);
    free(result_h);

    return 0;
}