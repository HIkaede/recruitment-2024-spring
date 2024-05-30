#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <limits>
#include <omp.h>

void optimized_pre_phase1(size_t) {}

void optimized_post_phase1() {}

void optimized_pre_phase2(size_t) {}

void optimized_post_phase2() {}

void optimized_do_phase1(float *data, size_t size)
{
    const size_t bnum = std::min((size_t)16, size);

    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < size; i++)
    {
        if (data[i] > max_val)
            max_val = data[i];
        if (data[i] < min_val)
            min_val = data[i];
    }

    std::vector<std::vector<float>> buckets(bnum + 1);
    float div = (max_val - min_val) / bnum;
    for (size_t i = 0; i < size; i++)
    {
        size_t index = (size_t)((data[i] - min_val) / div);
        buckets[index].push_back(data[i]);
    }

#pragma omp parallel for
    for (size_t i = 0; i < buckets.size(); i++)
    {
        if (!buckets[i].empty())
        {
            std::sort(buckets[i].begin(), buckets[i].end());
        }
    }

    size_t k = 0;
    for (const auto &bucket : buckets)
    {
        std::copy(bucket.begin(), bucket.end(), data + k);
        k += bucket.size();
    }
}

void optimized_do_phase2(size_t *result, float *data, float *query, size_t size)
{
#pragma omp parallel for
    for (size_t i = 0; i < size; ++i)
    {
        size_t l = 0, r = size;
        while (l < r)
        {
            size_t m = l + (r - l) / 2;
            if (data[m] < query[i])
            {
                l = m + 1;
            }
            else
            {
                r = m;
            }
        }
        result[i] = l;
    }
}
