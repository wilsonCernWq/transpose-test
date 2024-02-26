/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include <sycl/sycl.hpp>
#undef SYCL_EXT_ONEAPI_BINDLESS_IMAGES
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <assert.h>
#include <chrono>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int NUM_REPS = 100;

// Check errors and print GB/s
void postprocess(const float *ref, const float *res, int n, float ms)
{
  bool passed = true;
  for (int i = 0; i < n; i++)
    if (res[i] != ref[i]) {
      printf("\n");
      printf("#%d %f %f\n", i, res[i], ref[i]);
      printf("%25s\n", "*** FAILED ***");
      passed = false;
      break;
    }
  if (passed)
    printf("%20.2f\n", 2 * n * sizeof(float) * 1e-6 * NUM_REPS / ms );
}

// simple copy kernel
// Used as reference case representing best effective bandwidth.
void copy(float *odata, const float *idata, const sycl::nd_item<3> &item_ct1)
{
  int x = item_ct1.get_group(2) * TILE_DIM + item_ct1.get_local_id(2);
  int y = item_ct1.get_group(1) * TILE_DIM + item_ct1.get_local_id(1);
  int width = item_ct1.get_group_range(2) * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    odata[(y+j)*width + x] = idata[(y+j)*width + x];
}

// copy kernel using shared memory
// Also used as reference case, demonstrating effect of using shared memory.
void copySharedMem(float *odata, const float *idata,
                   const sycl::nd_item<3> &item_ct1, float *tile)
{

  int x = item_ct1.get_group(2) * TILE_DIM + item_ct1.get_local_id(2);
  int y = item_ct1.get_group(1) * TILE_DIM + item_ct1.get_local_id(1);
  int width = item_ct1.get_group_range(2) * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[(item_ct1.get_local_id(1) + j) * TILE_DIM +
          item_ct1.get_local_id(2)] = idata[(y + j) * width + x];

  item_ct1.barrier(sycl::access::fence_space::local_space);

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y + j) * width + x] =
         tile[(item_ct1.get_local_id(1) + j) * TILE_DIM +
              item_ct1.get_local_id(2)];
}

// naive transpose
// Simplest transpose; doesn't use shared memory.
// Global memory reads are coalesced but writes are not.
void transposeNaive(float *odata, const float *idata,
                    const sycl::nd_item<3> &item_ct1)
{
  int x = item_ct1.get_group(2) * TILE_DIM + item_ct1.get_local_id(2);
  int y = item_ct1.get_group(1) * TILE_DIM + item_ct1.get_local_id(1);
  int width = item_ct1.get_group_range(2) * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    odata[x*width + (y+j)] = idata[(y+j)*width + x];
}

// coalesced transpose
// Uses shared memory to achieve coalesing in both reads and writes
// Tile width == #banks causes shared memory bank conflicts.
void transposeCoalesced(float *odata, const float *idata,
                        const sycl::nd_item<3> &item_ct1,
                        sycl::local_accessor<float, 2> tile)
{
  int x = item_ct1.get_group(2) * TILE_DIM + item_ct1.get_local_id(2);
  int y = item_ct1.get_group(1) * TILE_DIM + item_ct1.get_local_id(1);
  int width = item_ct1.get_group_range(2) * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    tile[item_ct1.get_local_id(1) + j][item_ct1.get_local_id(2)] = idata[(y + j) * width + x];

  item_ct1.barrier(sycl::access::fence_space::local_space);

  x = item_ct1.get_group(1) * TILE_DIM + item_ct1.get_local_id(2); // transpose block offset
  y = item_ct1.get_group(2) * TILE_DIM + item_ct1.get_local_id(1);

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    odata[(y + j) * width + x] = tile[item_ct1.get_local_id(2)][item_ct1.get_local_id(1) + j];
}

// No bank-conflict transpose
// Same as transposeCoalesced except the first tile dimension is padded 
// to avoid shared memory bank conflicts.
void transposeNoBankConflicts(float *odata, const float *idata,
                              const sycl::nd_item<3> &item_ct1,
                              sycl::local_accessor<float, 2> tile)
{
  int x = item_ct1.get_group(2) * TILE_DIM + item_ct1.get_local_id(2);
  int y = item_ct1.get_group(1) * TILE_DIM + item_ct1.get_local_id(1);
  int width = item_ct1.get_group_range(2) * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    tile[item_ct1.get_local_id(1) + j][item_ct1.get_local_id(2)] = idata[(y + j) * width + x];

  item_ct1.barrier(sycl::access::fence_space::local_space);

  x = item_ct1.get_group(1) * TILE_DIM + item_ct1.get_local_id(2); // transpose block offset
  y = item_ct1.get_group(2) * TILE_DIM + item_ct1.get_local_id(1);

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    odata[(y + j) * width + x] = tile[item_ct1.get_local_id(2)][item_ct1.get_local_id(1) + j];
}

int main(int argc, char **argv)
{
  const int nx = 1024;
  const int ny = 1024;
  const int mem_size = nx*ny*sizeof(float);

  sycl::range<3> dimGrid(1, ny / TILE_DIM, nx / TILE_DIM);
  sycl::range<3> dimBlock(1, BLOCK_ROWS, TILE_DIM);

  int devId = 0;
  if (argc > 1) devId = atoi(argv[1]);

  dpct::device_info prop;
  DPCT_CHECK_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(devId)));
  printf("\nDevice : %s\n", prop.get_name());
  printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n", 
         nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
  printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n", 
         (int)dimGrid[2],  (int)dimGrid[1],  (int)dimGrid[0], 
         (int)dimBlock[2], (int)dimBlock[1], (int)dimBlock[0]);

  int max_work_group_size = dpct::dev_mgr::instance().current_device().get_info<sycl::info::device::max_work_group_size>();
  printf("max_work_group_size: %d\n", max_work_group_size);

  /*
  DPCT1093:10: The "devId" device may be not the one intended for use. Adjust
  the selected device if needed.
  */
  DPCT_CHECK_ERROR(dpct::select_device(devId));

  float *h_idata = (float*)malloc(mem_size);
  float *h_cdata = (float*)malloc(mem_size);
  float *h_tdata = (float*)malloc(mem_size);
  float *gold    = (float*)malloc(mem_size);
  
  float *d_idata, *d_cdata, *d_tdata;
  DPCT_CHECK_ERROR(d_idata = (float *)sycl::malloc_device(mem_size, dpct::get_in_order_queue()));
  DPCT_CHECK_ERROR(d_cdata = (float *)sycl::malloc_device(mem_size, dpct::get_in_order_queue()));
  DPCT_CHECK_ERROR(d_tdata = (float *)sycl::malloc_device(mem_size, dpct::get_in_order_queue()));

  // check parameters and calculate execution configuration
  if ((nx % TILE_DIM) | (ny % TILE_DIM)) {
    printf("nx and ny must be a multiple of TILE_DIM\n");
    throw std::runtime_error("Error");
  }

  if (TILE_DIM % BLOCK_ROWS) {
    printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
    throw std::runtime_error("Error");
  }
    
  // host
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      h_idata[j*nx + i] = j*nx + i;

  // correct result for error checking
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      gold[j*nx + i] = h_idata[i*nx + j];
  
  // device
  sycl::queue &queue = dpct::get_in_order_queue();
  DPCT_CHECK_ERROR(queue.memcpy(d_idata, h_idata, mem_size).wait());

  // events for timing
  dpct::event_ptr startEvent, stopEvent;
  std::chrono::time_point<std::chrono::steady_clock> startEvent_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stopEvent_ct1;
  startEvent = new sycl::event();
  stopEvent = new sycl::event();
  float ms;

  // ------------
  // time kernels
  // ------------
  
  printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");

  // ----
  // copy 
  // ----
  printf("%25s", "copy");
  DPCT_CHECK_ERROR(queue.memset(d_cdata, 0, mem_size).wait());
  // warm up
  /*
  DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  queue.parallel_for(
      sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
      [=](sycl::nd_item<3> item_ct1) {
        copy(d_cdata, d_idata, item_ct1);
      }).wait_and_throw();

  startEvent_ct1 = std::chrono::steady_clock::now();
  for (int i = 0; i < NUM_REPS; i++) {
     /*
     DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
     limit. To get the device limit, query info::device::max_work_group_size.
     Adjust the work-group size if needed.
     */
    *stopEvent = queue.parallel_for(
        sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
        [=](sycl::nd_item<3> item_ct1) {
          copy(d_cdata, d_idata, item_ct1);
        });
  }
  stopEvent->wait_and_throw();

  stopEvent_ct1 = std::chrono::steady_clock::now();
  ms = std::chrono::duration<float, std::milli>(stopEvent_ct1 - startEvent_ct1).count();
  DPCT_CHECK_ERROR(queue.memcpy(h_cdata, d_cdata, mem_size).wait());
  postprocess(h_idata, h_cdata, nx*ny, ms);

  // -------------
  // copySharedMem 
  // -------------
  printf("%25s", "shared memory copy");
  DPCT_CHECK_ERROR(queue.memset(d_cdata, 0, mem_size).wait());
  // warm up
  /*
  DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  queue.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> tile_acc_ct1(sycl::range<1>(TILE_DIM * TILE_DIM), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
        [=](sycl::nd_item<3> item_ct1) {
          copySharedMem(
              d_cdata, d_idata, item_ct1,
              tile_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get()
          );
        });
    }).wait_and_throw();

  startEvent_ct1 = std::chrono::steady_clock::now();
  for (int i = 0; i < NUM_REPS; i++) {
    *stopEvent = queue.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<float, 1> tile_acc_ct1(sycl::range<1>(TILE_DIM * TILE_DIM), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
          [=](sycl::nd_item<3> item_ct1) {
            copySharedMem(
                d_cdata, d_idata, item_ct1,
                tile_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get()
            );
          });
      });
  }
  stopEvent->wait_and_throw();

  stopEvent_ct1 = std::chrono::steady_clock::now();
  ms = std::chrono::duration<float, std::milli>(stopEvent_ct1 - startEvent_ct1).count();
  DPCT_CHECK_ERROR(queue.memcpy(h_cdata, d_cdata, mem_size).wait());
  postprocess(h_idata, h_cdata, nx * ny, ms);

  // --------------
  // transposeNaive 
  // --------------
  printf("%25s", "naive transpose");
  DPCT_CHECK_ERROR(queue.memset(d_tdata, 0, mem_size).wait());
  // warmup
  /*
  DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  queue.parallel_for(
      sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
      [=](sycl::nd_item<3> item_ct1) {
        transposeNaive(d_tdata, d_idata, item_ct1);
      }).wait_and_throw();

  startEvent_ct1 = std::chrono::steady_clock::now();
  for (int i = 0; i < NUM_REPS; i++) {
     /*
     DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the
     limit. To get the device limit, query info::device::max_work_group_size.
     Adjust the work-group size if needed.
     */
    *stopEvent = queue.parallel_for(
        sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
        [=](sycl::nd_item<3> item_ct1) {
          transposeNaive(d_tdata, d_idata, item_ct1);
        });
  }
  stopEvent->wait_and_throw();

  stopEvent_ct1 = std::chrono::steady_clock::now();
  ms = std::chrono::duration<float, std::milli>(stopEvent_ct1 - startEvent_ct1).count();
  DPCT_CHECK_ERROR(queue.memcpy(h_tdata, d_tdata, mem_size).wait());
  postprocess(gold, h_tdata, nx * ny, ms);

  // ------------------
  // transposeCoalesced 
  // ------------------
  printf("%25s", "coalesced transpose");
  DPCT_CHECK_ERROR(queue.memset(d_tdata, 0, mem_size).wait());
  // warmup
  /*
  DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  queue.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 2> tile_acc_ct1(sycl::range<2>(TILE_DIM, TILE_DIM), cgh);

    cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                     [=](sycl::nd_item<3> item_ct1) {
                       transposeCoalesced(d_tdata, d_idata, item_ct1, tile_acc_ct1);
                     });
    }).wait_and_throw();

  startEvent_ct1 = std::chrono::steady_clock::now();
  for (int i = 0; i < NUM_REPS; i++) {
     /*
     DPCT1049:8: The work-group size passed to the SYCL kernel may exceed the
     limit. To get the device limit, query info::device::max_work_group_size.
     Adjust the work-group size if needed.
     */
    *stopEvent = queue.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<float, 2> tile_acc_ct1(sycl::range<2>(TILE_DIM, TILE_DIM), cgh);

      cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                       [=](sycl::nd_item<3> item_ct1) {
                         transposeCoalesced(d_tdata, d_idata, item_ct1, tile_acc_ct1);
                       });
      });
  }
  stopEvent->wait_and_throw();

  stopEvent_ct1 = std::chrono::steady_clock::now();
  ms = std::chrono::duration<float, std::milli>(stopEvent_ct1 - startEvent_ct1).count();
  DPCT_CHECK_ERROR(queue.memcpy(h_tdata, d_tdata, mem_size).wait());
  postprocess(gold, h_tdata, nx * ny, ms);

  // ------------------------
  // transposeNoBankConflicts
  // ------------------------
  printf("%25s", "conflict-free transpose");
  DPCT_CHECK_ERROR(queue.memset(d_tdata, 0, mem_size).wait());
  // warmup
  /*
  DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  queue.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 2> tile_acc_ct1(sycl::range<2>(TILE_DIM, TILE_DIM+1), cgh);

    cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                     [=](sycl::nd_item<3> item_ct1) {
                       transposeNoBankConflicts(d_tdata, d_idata, item_ct1, tile_acc_ct1);
                     });
  }).wait_and_throw();

  startEvent_ct1 = std::chrono::steady_clock::now();
  for (int i = 0; i < NUM_REPS; i++) {
     /*
     DPCT1049:9: The work-group size passed to the SYCL kernel may exceed the
     limit. To get the device limit, query info::device::max_work_group_size.
     Adjust the work-group size if needed.
     */
    *stopEvent = queue.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<float, 2> tile_acc_ct1(sycl::range<2>(TILE_DIM, TILE_DIM+1), cgh);

      cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                       [=](sycl::nd_item<3> item_ct1) {
                         transposeNoBankConflicts(d_tdata, d_idata, item_ct1, tile_acc_ct1);
                       });
    });
  }
  stopEvent->wait_and_throw();

  stopEvent_ct1 = std::chrono::steady_clock::now();
  ms = std::chrono::duration<float, std::milli>(stopEvent_ct1 - startEvent_ct1).count();
  DPCT_CHECK_ERROR(queue.memcpy(h_tdata, d_tdata, mem_size).wait());
  postprocess(gold, h_tdata, nx * ny, ms);

  // cleanup
  DPCT_CHECK_ERROR(dpct::destroy_event(startEvent));
  DPCT_CHECK_ERROR(dpct::destroy_event(stopEvent));
  DPCT_CHECK_ERROR(dpct::dpct_free(d_tdata, dpct::get_in_order_queue()));
  DPCT_CHECK_ERROR(dpct::dpct_free(d_cdata, dpct::get_in_order_queue()));
  DPCT_CHECK_ERROR(dpct::dpct_free(d_idata, dpct::get_in_order_queue()));
  free(h_idata);
  free(h_tdata);
  free(h_cdata);
  free(gold);
}
