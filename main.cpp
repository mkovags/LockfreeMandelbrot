#include <iostream>
#include <thread>
#include <atomic>
#include <vector>

using ResultType = uint16_t;

const auto cBatchSize{20000};
static constexpr int cWidthPixels{170};
static constexpr int cHeightPixels{118};

/**
 * Each thread keeps a private buffer.
 * After calculating each iteration, it saves the result to the buffer.
 * Afterwards it checks an atomic flag to tell if another thread is already saving their results.
 * If no thread is saving its results, it immediately swaps the flag to true
 * and starts to dump its buffer contents into the "main memory".
 * If someone was already reading, that's ok because the buffer is large and the thread will try to dump next time
 * Atomic variable access is done with the acquire / release
 * semantics, which gives more opportunities for the compiler
 * and cpu to optimize
 */
class Stack {
private:
    static constexpr int cBufferSize{99999};
    struct Item {
        int x;
        int y;
        ResultType res;
    };

    std::array<Item, cBufferSize> arr;
    int lastPos{0};

public:
    void push(Item item) {
        arr[lastPos] = item;
        ++lastPos;
    }

    void dumpToMap(std::vector<std::vector<ResultType>> &resultMap) {
        for (int i = 0; i != lastPos; ++i) {
            const auto &e = arr[i];
            resultMap[e.x][e.y] = e.res;
        }
        lastPos = 0;
    }
};

class LockfreeMandelbrot {
public:
    struct MandelbrotBitmap {
        MandelbrotBitmap(int width, int height) : width{width}, height{height} {
            map.emplace_back();

            map[0].reserve(height);
            for (int y = 0; y != height; ++y) {
                map[0].emplace_back();
            }

            map.reserve(width);
            for (int x = 1; x != width; ++x) {
                map.push_back(map[0]);
            }
        }

        std::vector<std::vector<ResultType>> map;
        int width;
        int height;
    };

    LockfreeMandelbrot(int width, int height)
            : mWidthPixels{width},
              mHeightPixels{height},
              mResults{width, height},
              mIsStarted{false} {}

    ~LockfreeMandelbrot() {
        waitToFinish();
    }

    void waitToFinish() {
        if (mIsStarted == false) {
            return;
        }
        mIsStarted.store(false);

        for (auto &t: mThreads) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    bool startThreads(int numThreads) {
        mMandelbrotIterator.store(0);
        bool expected{false};
        if (mIsStarted.compare_exchange_strong(expected, true)) {
            mThreads.reserve(numThreads);
            for (int i = 0; i != numThreads; ++i) {
                mThreads.emplace_back(std::thread([this]() {
                    loop();
                }));
            }
            return true;
        } else {
            return false;
        }
    }

    const MandelbrotBitmap &getMap() const {
        return mResults;
    }

private:
    double scaleX(uint32_t x) {
        return static_cast<double>(x) / mWidthPixels * 2.47 - 2;
    }

    double scaleY(uint32_t y) {
        return static_cast<double>(y) / mHeightPixels * 2.24 - 1.12;
    }

    void loop() {
        Stack threadStack;

        const auto totalArea{mWidthPixels * mHeightPixels};

        int currentBatch, currentIndex{-1};
        while (true) {
            ++currentIndex;
            if(currentIndex % cBatchSize == 0 || currentIndex >= totalArea) {
                bool dummy{false};
                while (!mIsSavingResults.compare_exchange_weak(dummy, true, std::memory_order_acquire,
                                                               std::memory_order_relaxed));
                threadStack.dumpToMap(mResults.map);
                mIsSavingResults.store(false, std::memory_order_release);

                currentBatch = mMandelbrotIterator.fetch_add(1);
                if(currentBatch >= totalArea / cBatchSize ) {
                    return;
                }

                currentIndex = currentBatch * cBatchSize;
            }

            int x, y;
            positionFromIndex(currentIndex, x, y);

            auto scaledX{scaleX(x)};
            auto scaledY{scaleY(y)};

            auto result = render(scaledX, scaledY);
            threadStack.push({x, y, result});
        }
    }

    ResultType render(const double &scaledX, const double &scaledY) {
        double x{0};
        double y{0};
        int i = 0;
        for (; i != cMaxIterations; ++i) {
            double xtemp = x * x - y * y + scaledX;
            y = 2 * x * y + scaledY;
            x = xtemp;
            if (x * x + y * y > 2 * 2) {
                break;
            }
        }
        return i;
    }

    /**
     * Returns the X,Y coordinates from an index between width x height
     * @param in The index
     * @param x
     * @param y
     */
    void positionFromIndex(const uint64_t in, int &x, int &y) {
        y = in / mWidthPixels;
        x = in - (y * mWidthPixels);
    }

    static constexpr int cMaxIterations{1000};
    std::vector<std::thread> mThreads;
    std::atomic<uint64_t> mMandelbrotIterator;
    std::atomic<bool> mIsSavingResults;
    MandelbrotBitmap mResults;
    std::atomic<bool> mIsStarted;
    int mWidthPixels;
    int mHeightPixels;
};

void drawFromResults(const LockfreeMandelbrot::MandelbrotBitmap &mandelbrotBitmap) {
    for (int y = 0; y != mandelbrotBitmap.height; ++y) {
        for (int x = 0; x != mandelbrotBitmap.width; ++x) {
            auto r = mandelbrotBitmap.map[x][y];
            char c;
            if (r <= 10) {
                c = ' ';
            } else if (r > 10 && r <= 100) {
                c = '.';
            } else if (r > 100 && r <= 200) {
                c = 'x';
            } else if (r > 200 && r <= 1000) {
                c = 'O';
            }
            std::cout << c;
        }
        std::cout << "\n";
    }
}

int main() {
    LockfreeMandelbrot lfm(cWidthPixels, cHeightPixels);

    auto t1 = std::chrono::high_resolution_clock::now();

    lfm.startThreads(24);
    lfm.waitToFinish();
    drawFromResults(lfm.getMap());

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Calculation took: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1000.0f << "s to complete\n";

    return 0;
}
