#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct Effect {
    std::array<uint64_t, 6> d{};
    uint64_t o = 0;
};

struct PairRecord {
    uint64_t hash = 0;
    uint64_t pair = 0;
    uint64_t o = 0;
};

struct DetectorHash {
    std::size_t operator()(const std::array<uint64_t, 6>& value) const;
};

static uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static uint64_t hash_detector(const std::array<uint64_t, 6>& value) {
    uint64_t h = 0x243f6a8885a308d3ULL;
    for (unsigned k = 0; k < 6; ++k) {
        h ^= mix(value[k] + 0x9e3779b97f4a7c15ULL * (k + 1));
        h = (h << 17) | (h >> 47);
    }
    return mix(h);
}

std::size_t DetectorHash::operator()(const std::array<uint64_t, 6>& value) const {
    return static_cast<std::size_t>(hash_detector(value));
}

static std::array<uint64_t, 6> pair_detector(
    const std::vector<Effect>& effects, uint64_t packed
) {
    const uint32_t i = static_cast<uint32_t>(packed >> 32);
    const uint32_t j = static_cast<uint32_t>(packed);
    std::array<uint64_t, 6> output{};
    for (unsigned k = 0; k < 6; ++k) output[k] = effects[i].d[k] ^ effects[j].d[k];
    return output;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: pair_certify_lp160 EFFECTS.bin\n";
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) throw std::runtime_error("cannot open input");
    uint32_t n = 0, detectors = 0, observables = 0, words = 0;
    in.read(reinterpret_cast<char*>(&n), 4);
    in.read(reinterpret_cast<char*>(&detectors), 4);
    in.read(reinterpret_cast<char*>(&observables), 4);
    in.read(reinterpret_cast<char*>(&words), 4);
    if (words != 6 || detectors != 288 || observables != 16) {
        throw std::runtime_error("unexpected signature dimensions");
    }
    std::vector<Effect> effects(n);
    for (auto& effect : effects) {
        in.read(reinterpret_cast<char*>(effect.d.data()), 6 * sizeof(uint64_t));
        in.read(reinterpret_cast<char*>(&effect.o), sizeof(uint64_t));
    }
    if (!in) throw std::runtime_error("truncated input");
    std::unordered_map<std::array<uint64_t, 6>, std::vector<uint32_t>, DetectorHash>
        effects_by_detector;
    for (uint32_t index = 0; index < n; ++index) {
        bool zero = true;
        for (auto word : effects[index].d) zero &= word == 0;
        if (zero && effects[index].o) {
            std::cout << "{\"verified\":false,\"faults\":1}\n";
            return 1;
        }
        effects_by_detector[effects[index].d].push_back(index);
    }
    const uint64_t pair_count = static_cast<uint64_t>(n) * (n - 1) / 2;
    std::cerr << "effects=" << n << " pairs=" << pair_count << " stage=generate\n";
    std::vector<PairRecord> pairs;
    pairs.reserve(pair_count);
    std::array<uint64_t, 6> d{};
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = i + 1; j < n; ++j) {
            for (unsigned k = 0; k < 6; ++k) d[k] = effects[i].d[k] ^ effects[j].d[k];
            const uint64_t observable = effects[i].o ^ effects[j].o;
            bool zero = true;
            for (auto word : d) zero &= word == 0;
            if (zero && observable) {
                std::cout << "{\"verified\":false,\"faults\":2}\n";
                return 1;
            }
            auto singles = effects_by_detector.find(d);
            if (singles != effects_by_detector.end()) {
                for (uint32_t third : singles->second) {
                    if (third != i && third != j && (observable ^ effects[third].o)) {
                        std::cout << "{\"verified\":false,\"faults\":3,\"indices\":["
                                  << i << ',' << j << ',' << third << "]}\n";
                        return 1;
                    }
                }
            }
            pairs.push_back({hash_detector(d), (static_cast<uint64_t>(i) << 32) | j, observable});
        }
    }
    std::cerr << "stage=sort bytes=" << pairs.size() * sizeof(PairRecord) << "\n";
    std::sort(pairs.begin(), pairs.end(), [](const PairRecord& a, const PairRecord& b) {
        return a.hash < b.hash;
    });
    std::cerr << "stage=scan\n";
    uint64_t collision_buckets = 0;
    uint64_t maximum_bucket = 0;
    for (uint64_t begin = 0; begin < pairs.size();) {
        uint64_t end = begin + 1;
        while (end < pairs.size() && pairs[end].hash == pairs[begin].hash) ++end;
        maximum_bucket = std::max(maximum_bucket, end - begin);
        if (end - begin > 1) {
            ++collision_buckets;
            struct Representative {
                std::array<uint64_t, 6> detector;
                uint64_t observable;
                uint64_t pair;
            };
            std::vector<Representative> representatives;
            for (uint64_t position = begin; position < end; ++position) {
                auto detector = pair_detector(effects, pairs[position].pair);
                bool matched = false;
                for (const auto& representative : representatives) {
                    if (detector == representative.detector) {
                        matched = true;
                        if (pairs[position].o != representative.observable) {
                            uint32_t a = representative.pair >> 32;
                            uint32_t b = static_cast<uint32_t>(representative.pair);
                            uint32_t c = pairs[position].pair >> 32;
                            uint32_t e = static_cast<uint32_t>(pairs[position].pair);
                            std::cout << "{\"verified\":false,\"faults\":4,\"indices\":["
                                      << a << ',' << b << ',' << c << ',' << e << "]}\n";
                            return 1;
                        }
                        break;
                    }
                }
                if (!matched) representatives.push_back({detector, pairs[position].o, pairs[position].pair});
            }
        }
        begin = end;
    }
    std::cout << "{\"verified\":true,\"no_faults_at_most_4\":true,\"effects\":"
              << n << ",\"pairs\":" << pair_count << ",\"hash_collision_buckets\":"
              << collision_buckets << ",\"maximum_hash_bucket\":" << maximum_bucket << "}\n";
    return 0;
}
