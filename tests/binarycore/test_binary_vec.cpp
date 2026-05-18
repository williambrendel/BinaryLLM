#include "binarycore/binary_vec.hpp"
#include "doctest.h"

using namespace binarycore;

TEST_CASE("BinaryVec default-constructs to all zeros") {
    BinaryVec64 v;
    CHECK(v.popcount() == 0);
    for (std::size_t i = 0; i < 64; ++i) {
        CHECK_FALSE(v.get(i));
    }
}

TEST_CASE("BinaryVec set/get/clear individual bits") {
    BinaryVec128 v;
    v.set(0);
    v.set(63);
    v.set(64);
    v.set(127);
    CHECK(v.get(0));
    CHECK(v.get(63));
    CHECK(v.get(64));
    CHECK(v.get(127));
    CHECK_FALSE(v.get(1));
    CHECK_FALSE(v.get(126));
    CHECK(v.popcount() == 4);

    v.clear(63);
    CHECK_FALSE(v.get(63));
    CHECK(v.popcount() == 3);
}

TEST_CASE("BinaryVec assign() handles both true and false") {
    BinaryVec64 v;
    v.assign(5, true);
    CHECK(v.get(5));
    v.assign(5, false);
    CHECK_FALSE(v.get(5));
}

TEST_CASE("BinaryVec bitwise AND / OR / XOR / NOT") {
    BinaryVec64 a;
    BinaryVec64 b;
    a.set(0); a.set(1); a.set(2);
    b.set(1); b.set(2); b.set(3);

    auto and_ = a & b;
    CHECK(and_.popcount() == 2);
    CHECK(and_.get(1));
    CHECK(and_.get(2));

    auto or_ = a | b;
    CHECK(or_.popcount() == 4);
    CHECK(or_.get(0));
    CHECK(or_.get(3));

    auto xor_ = a ^ b;
    CHECK(xor_.popcount() == 2);
    CHECK(xor_.get(0));
    CHECK(xor_.get(3));
    CHECK_FALSE(xor_.get(1));

    BinaryVec64 zero;
    auto not_zero = ~zero;
    CHECK(not_zero.popcount() == 64);
}

TEST_CASE("BinaryVec in-place bitwise operations") {
    BinaryVec64 a;
    a.set(0); a.set(1);
    BinaryVec64 b;
    b.set(1); b.set(2);

    BinaryVec64 c = a;
    c &= b;
    CHECK(c.popcount() == 1);
    CHECK(c.get(1));

    c = a;
    c |= b;
    CHECK(c.popcount() == 3);

    c = a;
    c ^= b;
    CHECK(c.popcount() == 2);
}

TEST_CASE("BinaryVec equality") {
    BinaryVec128 a, b;
    a.set(0); a.set(99);
    b.set(0); b.set(99);
    CHECK(a == b);
    CHECK_FALSE(a != b);
    b.set(50);
    CHECK_FALSE(a == b);
    CHECK(a != b);
}

TEST_CASE("BinaryVec Hamming distance") {
    BinaryVec64 a, b;
    CHECK(a.hamming(b) == 0); // both zero

    a.set(0); a.set(1); a.set(2);
    b.set(2); b.set(3); b.set(4);
    // a XOR b = {0,1,3,4} → hamming = 4
    CHECK(a.hamming(b) == 4);
    CHECK(b.hamming(a) == 4); // symmetric

    // Max distance: all-zero vs all-ones
    BinaryVec64 ones = ~BinaryVec64();
    CHECK(BinaryVec64().hamming(ones) == 64);
}

TEST_CASE("BinaryVec dot product (popcount of AND)") {
    BinaryVec64 a, b;
    a.set(0); a.set(1); a.set(2);
    b.set(2); b.set(3); b.set(4);
    // a AND b = {2} → dot = 1
    CHECK(a.dot(b) == 1);

    // dot(x, x) == popcount(x)
    CHECK(a.dot(a) == 3);
    CHECK(b.dot(b) == 3);

    // dot(x, 0) == 0
    CHECK(a.dot(BinaryVec64()) == 0);
}

TEST_CASE("BinaryVec similarity in [0,1]") {
    BinaryVec64 a, b;
    CHECK(a.similarity(b) == doctest::Approx(1.0f)); // identical (both zero)

    a.set(0);
    CHECK(a.similarity(b) == doctest::Approx(1.0f - 1.0f / 64.0f));

    BinaryVec64 ones = ~BinaryVec64();
    CHECK(BinaryVec64().similarity(ones) == doctest::Approx(0.0f)); // opposite
}

TEST_CASE("BinaryVec sizes: 64, 128, 256, 512") {
    CHECK(BinaryVec64::Dims == 64);
    CHECK(BinaryVec64::Chunks == 1);
    CHECK(BinaryVec128::Dims == 128);
    CHECK(BinaryVec128::Chunks == 2);
    CHECK(BinaryVec256::Dims == 256);
    CHECK(BinaryVec256::Chunks == 4);
    CHECK(BinaryVec512::Dims == 512);
    CHECK(BinaryVec512::Chunks == 8);
}

TEST_CASE("BinaryVec construction from explicit chunks") {
    BinaryVec128 v({0xFFFFFFFFFFFFFFFFULL, 0x0ULL});
    CHECK(v.popcount() == 64);
    CHECK(v.get(0));
    CHECK(v.get(63));
    CHECK_FALSE(v.get(64));
}
