#include "engine/community_models/moss_voicegen/assets.h"

#include "engine/framework/model_spec/package.h"

#include <utility>

namespace engine::models::moss_voicegen {

namespace {
constexpr const char * kFamilyName = "MOSS-VoiceGenerator";
}

std::shared_ptr<const MossVoiceGenAssets> load_moss_voicegen_assets(const std::filesystem::path & model_path) {
    MossVoiceGenAssets assets;
    assets.resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("moss_voicegen"));
    assets.config = delay::parse_config(assets.resources, kFamilyName);
    assets.model_weights = assets.resources.open_tensor_source("model_weights");
    assets.audio_tokenizer_weights = assets.resources.open_tensor_source("audio_tokenizer_weights");
    return std::make_shared<MossVoiceGenAssets>(std::move(assets));
}

}  // namespace engine::models::moss_voicegen
