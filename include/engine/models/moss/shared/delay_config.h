#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/io/json.h"

#include <cstdint>

namespace engine::models::moss::delay {

// Qwen3 backbone geometry, read from the checkpoint's "language_config" block. Every
// moss_tts_delay checkpoint carries this block; only the sizes differ (VoiceGenerator is
// 1.7B, MOSS-TTSD and MOSS-TTS are 8B).
struct BackboneConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t max_position_embeddings = 0;
    int64_t vocab_size = 0;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 1000000.0F;
    bool tie_word_embeddings = true;
};

// The moss_tts_delay family emits 1 + n_vq ids per step: one text id and one code per
// codebook. n_vq is 16 across the released checkpoints, half of what the architecture
// allows.
struct Config {
    BackboneConfig backbone;
    int64_t num_codebooks = 0;
    int64_t audio_vocab_size = 0;
    int64_t audio_pad_code = 0;
    // The checkpoints omit these; MossTTSDelayConfig's defaults apply and match what the
    // tokenizer resolves for <|im_start|>/<|im_end|>.
    int64_t pad_token_id = 151643;
    int64_t im_start_token_id = 151644;
    int64_t im_end_token_id = 151645;
    int64_t audio_start_token_id = 0;
    int64_t audio_end_token_id = 0;
    int64_t audio_user_slot_token_id = 0;
    int64_t audio_assistant_gen_slot_token_id = 0;
    int64_t audio_assistant_delay_slot_token_id = 0;
    int64_t sampling_rate = 0;
};

// Parses a moss_tts_delay config.json. `family` only names the model in error messages.
Config parse_config(const engine::io::json::Value & root, const char * family);
Config parse_config(const assets::ResourceBundle & resources, const char * family);

}  // namespace engine::models::moss::delay
