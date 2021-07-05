#include <atomic>
#include <thread>
#include <vector>
#include <condition_variable>
#include <mutex>
#include <functional>
#include <chrono>
#include <iostream>
#include <random>

using namespace std;

class PrefixSumSolver {

    static const int NoT = 32;
    static const int BUFFER_SIZE = 2000;

    class ThreadPool {
        function<void(void)> queue[BUFFER_SIZE];

        thread t_pool[NoT];

        mutex c_lock;
        int pid = 0, cid = 0;
        condition_variable consumer_cv, producer_cv, waitall_cv;
        bool halt = false;

        public:
        ThreadPool() {
            for (int i = 0; i < NoT; ++i) {
                t_pool[i] = thread([&]() {
                    while(true) {
                        function<void(void)> task;
                        {
                            unique_lock<mutex> lock(c_lock);
                            if (halt) break;
                            consumer_cv.wait(lock, [&]{
                                int d = __cdis(pid,cid);
                                return halt || (d > 0 && d <= BUFFER_SIZE/2);
                            });
                            if (halt) return;
                            task = queue[cid];
                            if (__cdis(pid, cid) == BUFFER_SIZE/2) {
                                producer_cv.notify_all();
                            }
                            cid = (cid+1)%BUFFER_SIZE;
                        }
                        task();
                    } 
                });
            }
        }
        int __cdis(int a, int b) {
            return (a>=b)? a - b: a + BUFFER_SIZE - b;
        }

        void produce(function<void(void)> task) {
            unique_lock<mutex> lock(c_lock);
            producer_cv.wait(lock, [&]{
                int d = __cdis(pid,cid);
                return halt || (d >= 0 && d < BUFFER_SIZE/2);
            });
            if (halt) return;
            queue[pid] = task;
            if (pid == cid) {
                consumer_cv.notify_all();
            }
            pid = (pid+1)%BUFFER_SIZE;       

        }


        void waitAll() {
            unique_lock<mutex> lock(c_lock);
            if (pid == cid) return;
            waitall_cv.wait(lock, [&]{return pid == cid;});
        }

        ~ThreadPool() {
            {
                unique_lock<mutex> lock(c_lock);
                halt = true;
                consumer_cv.notify_all();
            }
            
            for (int i = 0; i < NoT; ++i) {
                if(t_pool[i].joinable()) {
                    t_pool[i].join();
                }
            }
        }
    };
    

    
public:


    PrefixSumSolver(){}

    ~PrefixSumSolver() {

    }

    void prefix_sum_inplace_multi(int * target, int N) {
        
        // upsweeping

        int offset = 1;
        for (int d = N >> 1; d > 0; d>>=1) {
            thread t[d];
            for (int i = 1; i <= d; ++i) {
                t[i-1] = thread([i, offset, target]() {
                    int a = offset*(2*i) - 1;
                    int b = offset*(2*i - 1) - 1;
                    target[a] += target[b]; 
                });
            }
            for (int i = 0; i < d; ++i) {
                t[i].join();
            }
            offset <<= 1;
        }
        // downsweeping

        
        int last = target[N - 1];
        target[N - 1] = 0;

        offset = N >> 1;
        for (int d = 1; d <= (N >> 1); d<<=1) {
            thread t[d];
            for (int i = 1; i <= d; ++i) {
                t[i-1] = thread([i, offset, target]() {
                    int a = offset*(2*i-1) - 1, b = offset*(2*i) - 1;
                    target[a] += target[b];
                    swap(target[a], target[b]);
                });
            }
            for (int i = 0; i < d; ++i) {
                t[i].join();
            }
            offset >>= 1;
        }

        // if wanted inplace, could do a parallel shift
        // offset = 1;
        // for (int d = N >> 1; d > 0; d>>=1) {
        //     for (int i = 1; i <= d; ++i) {
        //         pool.produce([i, offset, target]() {
        //             int a = offset*(2*i) - 1;
        //             int b = offset*(2*i - 1) - 1;
        //             swap(target[a], target[b]); 
        //         });
        //     }
        //     offset <<= 1;
        //     pool.waitAll();
        // }
        // target[N - 1] = last;

    }

    void prefix_sum_inplace_single(int * target, int N) {
        for (int i = 0; i < N-1; ++i) {
            target[i+1] += target[i];
        }
    }
};




int main() {
    const int test_num = 1 << 20;

    srand(0);

    PrefixSumSolver solver;
    // {
    //     int test[test_num];
    //     for (int i = 0; i < test_num; ++i) {
    //         test[i] = rand() %1000;
    //     }
    //     auto start = chrono::steady_clock::now();
    //     solver.prefix_sum_inplace_multi(test, test_num);
    //     auto end = chrono::steady_clock::now();

    //     auto diff = end - start;
    //     cout << chrono::duration <double, milli> (diff).count() << " ms" << endl;
    //     for (int i = 0; i < test_num; ++i) {
    //         cout << test[i] << " "; 
    //     }
    //     cout << endl;
    // }

    {
        int test[test_num];
        for (int i = 0; i < test_num; ++i) {
            test[i] = rand()%10;
        }
        auto start = chrono::steady_clock::now();
        solver.prefix_sum_inplace_single(test, test_num);
        auto end = chrono::steady_clock::now();

        auto diff = end - start;
        cout << chrono::duration <double, milli> (diff).count() << " ms" << endl;
        // for (int i = 0; i < test_num; ++i) {
        //     cout << test[i] << " "; 
        // }
        cout << endl;
    }

    return 0;
}