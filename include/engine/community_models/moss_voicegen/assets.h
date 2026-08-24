#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/models/moss/shared/delay_config.h"

#include <filesystem>
#include <memory>

namespace engine::models::moss_voicegen {

namespace delay = engine::models::moss::delay;

struct MossVoiceGenAssets {
    assets::ResourceBundle resources;
    // MOSS-VoiceGenerator is a moss_tts_delay checkpoint with n_vq 16, so the geometry and
    // the token ids come from the shared family config.
    delay::Config config;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> audio_tokenizer_weights;
};

std::shared_ptr<const MossVoiceGenAssets> load_moss_voicegen_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::moss_voicegen
