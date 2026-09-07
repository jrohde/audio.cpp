// Codec parity for MOSS-Audio-Tokenizer v1 in the encode direction (24 kHz mono waveform
// -> RLFQ codes). The delay family clones a speaker from a short reference recording, so
// the encoder has to be as right as the decoder; this compares the produced code matrix
// against tests/moss/reference/ref_codec_v1_encode.json, dumped from the checkpoint's own
// MossAudioTokenizerModel.
//
// The waveform is generated from the same formula as the reference dumper rather than read
// from a file, so the fixture needs no audio shipped alongside it.
//
// The codec weights are opened directly rather than through a model package, so this test
// stays independent of any one model in the family and can run against the f32 safetensors
// as well as against a converted package.
//
//   moss_codec_encode_parity --codec <model.safetensors.index.json|.gguf> \
//       --reference tests/moss/reference/ref_codec_v1_encode.json \
//       [--tensor-prefix audio_tokenizer_weights/]

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/json.h"
#include "engine/framework/codecs/moss_audio_tokenizer_codec_runtime.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace json = engine::io::json;

constexpr int64_t kSampleRate = 24000;
constexpr int64_t kSeconds = 4;
constexpr int64_t kQuantizers = 16;

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

// Mirrors waveform() in tools/community_models/moss_delay_reference_codec_encode.py.
// Three partials under a slow envelope: enough structure for the quantizer to latch onto,
// where noise would sit near the decision boundaries and make near-ties the norm.
std::vector<float> waveform() {
    const int64_t length = kSampleRate * kSeconds;
    std::vector<float> out(static_cast<size_t>(length));
    constexpr double kPi = 3.14159265358979323846;
    for (int64_t i = 0; i < length; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(kSampleRate);
        const double envelope = 0.5 + 0.4 * std::sin(2.0 * kPi * 0.7 * t);
        const double signal = 0.55 * std::sin(2.0 * kPi * 174.0 * t)
            + 0.28 * std::sin(2.0 * kPi * 349.0 * t + 0.6)
            + 0.13 * std::sin(2.0 * kPi * 523.0 * t + 1.3);
        out[static_cast<size_t>(i)] = static_cast<float>(envelope * signal * 0.7);
    }
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::string codec_path = arg_value(argc, argv, "--codec", "");
        const std::string reference_path = arg_value(argc, argv, "--reference", "");
        if (codec_path.empty() || reference_path.empty()) {
            std::cerr << "usage: moss_codec_encode_parity --codec <dir> --reference <json>\n";
            return 2;
        }

        const auto reference = json::parse_file(reference_path);
        const std::string tensor_prefix = arg_value(argc, argv, "--tensor-prefix", "");
        // A sharded checkpoint is addressed through its index; a converted package is a
        // single file. Both reach the same TensorSource interface.
        const std::filesystem::path codec_file(codec_path);
        auto weights = codec_file.filename() == "model.safetensors.index.json"
            ? engine::assets::open_indexed_tensor_source(codec_file, codec_file.parent_path())
            : engine::assets::open_tensor_source(codec_file);
        if (!tensor_prefix.empty()) {
            weights = engine::assets::make_prefixed_tensor_source(std::move(weights), tensor_prefix);
        }

        engine::core::BackendConfig backend_config;
        backend_config.type = engine::core::BackendType::Cpu;
        backend_config.device = 0;
        backend_config.threads = std::stoi(arg_value(argc, argv, "--threads", "8"));
        engine::core::ExecutionContext execution_context(backend_config);

        engine::codecs::MossAudioTokenizerCodecRuntimeOptions options;
        options.weight_context_bytes = 4096ull * 1024ull * 1024ull;
        options.encoder_graph_arena_bytes = 2048ull * 1024ull * 1024ull;
        engine::codecs::MossAudioTokenizerCodecRuntime codec(
            std::move(weights),
            execution_context,
            kQuantizers,
            options,
            engine::codecs::moss_audio_tokenizer_v1_config());

        engine::codecs::MossAudioTokenizerAudio input;
        input.sampling_rate = kSampleRate;
        input.channels = {waveform()};
        const auto result = codec.encode(input);
        const auto & codes = result.codebooks;

        const auto expected_quantizers = json::require_i64(reference, "quantizers");
        const auto expected_frames = json::require_i64(reference, "frames");
        std::cout << "codes=" << codes.size() << " x " << (codes.empty() ? 0 : codes.front().size())
                  << " (reference " << expected_quantizers << " x " << expected_frames << ")\n";
        if (static_cast<int64_t>(codes.size()) != expected_quantizers) {
            std::cerr << "FAIL: quantizer count does not match the reference\n";
            return 1;
        }

        const auto & rows = reference.require("codes").as_array();
        int64_t compared = 0;
        int64_t mismatched = 0;
        int64_t first_row_mismatched = 0;
        for (size_t q = 0; q < codes.size() && q < rows.size(); ++q) {
            const auto & row = rows[q].as_array();
            for (size_t t = 0; t < codes[q].size() && t < row.size(); ++t) {
                ++compared;
                if (codes[q][t] != static_cast<int32_t>(row[t].as_number())) {
                    ++mismatched;
                    if (q == 0) {
                        ++first_row_mismatched;
                    }
                }
            }
        }
        const double rate = compared == 0 ? 0.0 : 100.0 * static_cast<double>(compared - mismatched)
            / static_cast<double>(compared);
        std::cout << "matching codes=" << (compared - mismatched) << "/" << compared
                  << " (" << rate << "%), quantizer 0 mismatches=" << first_row_mismatched << "\n";

        if (mismatched != 0) {
            std::cerr << "FAIL: encoder codes differ from the reference\n";
            return 1;
        }
        std::cout << "PASS: encoder matches the reference code for code\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
