from __future__ import annotations

from dataclasses import dataclass


ANTI_RADIAL_NEGATIVE = (
    "eye, iris, pupil, eyeball, portal, vortex, tunnel, target symbol, "
    "concentric circles, repeated rings, circular artifact, ring pattern, "
    "bullseye, hypnotic circle, abstract circular texture"
)


@dataclass
class PromptVariant:
    label: str
    prompt: str
    negative_prompt: str


def normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").split()).strip()


def merge_negative_prompts(base_negative_prompt: str, extra_negative_prompt: str) -> str:
    seen_lower: set[str] = set()
    merged: list[str] = []

    for part in f"{base_negative_prompt}, {extra_negative_prompt}".split(","):
        cleaned = part.strip()
        if not cleaned:
            continue

        lowered = cleaned.lower()
        if lowered in seen_lower:
            continue

        seen_lower.add(lowered)
        merged.append(cleaned)

    return ", ".join(merged)


def suffixes_for_profile(
    evaluation_profile: str,
    portrait_reference_mode: bool = False,
) -> list[tuple[str, str, str]]:
    profile = (evaluation_profile or "portfolio").strip().lower()

    if profile == "portrait":
        return [
            (
                "portrait-clean",
                (
                    "Emphasize a clean realistic human portrait with natural facial proportions, "
                    "clear eyes, coherent face structure, realistic skin detail, and balanced lighting."
                ),
                "distorted face, extra eyes, deformed mouth, asymmetrical facial features, duplicated face",
            ),
            (
                "portrait-studio",
                (
                    "Emphasize a refined studio-quality portrait with natural expression, "
                    "professional lighting, clean background separation, and realistic anatomy."
                ),
                "surreal face, malformed nose, bad anatomy, extra limbs, duplicate face",
            ),
            (
                "portrait-minimal",
                (
                    "Emphasize a premium minimal portrait with a clear single subject, "
                    "simple background, polished lighting, and believable facial identity."
                ),
                "crowded background, multiple people, distorted facial geometry",
            ),
            (
                "portrait-closeup",
                (
                    "Emphasize a close-up portrait with stable identity, detailed eyes, "
                    "clean mouth shape, and realistic skin texture."
                ),
                "face artifacts, blurry face, extra facial elements, duplicated eyes",
            ),
        ]

    if profile == "reference_match":
        if portrait_reference_mode:
            return [
                (
                    "reference-portrait-identity",
                    (
                        "Emphasize faithful preservation of the reference person's face shape, "
                        "identity cues, hairstyle, gaze direction, and overall portrait likeness."
                    ),
                    "different person, face distortion, extra face, duplicated eyes, altered hairstyle",
                ),
                (
                    "reference-portrait-lighting",
                    (
                        "Emphasize preservation of the reference portrait lighting, camera angle, "
                        "framing, expression, and natural skin rendering."
                    ),
                    "wrong lighting, wrong expression, different angle, surreal face",
                ),
                (
                    "reference-portrait-clean",
                    (
                        "Emphasize a clean high-fidelity portrait reconstruction with strong likeness, "
                        "reduced artifacts, and stable facial structure."
                    ),
                    "messy reconstruction, noisy face, malformed eyes, malformed mouth",
                ),
                (
                    "reference-portrait-background",
                    (
                        "Emphasize preservation of the reference image mood and background feel "
                        "while keeping the portrait identity coherent and realistic."
                    ),
                    "unrelated background, multiple heads, background clutter",
                ),
            ]

        return [
            (
                "reference-structure",
                (
                    "Emphasize preservation of the reference image structure, composition, "
                    "main subject silhouette, and overall scene layout."
                ),
                "major composition change, unrelated subject, extreme framing change",
            ),
            (
                "reference-colors",
                (
                    "Emphasize preservation of the reference image color mood, "
                    "lighting atmosphere, and visual identity."
                ),
                "different palette, wrong lighting, unrelated aesthetic",
            ),
            (
                "reference-clean",
                (
                    "Emphasize a clean faithful reconstruction of the reference image "
                    "with strong visual similarity and reduced artifacts."
                ),
                "messy reconstruction, noisy image, deformed shapes, broken structure",
            ),
            (
                "reference-detail",
                (
                    "Emphasize preservation of the reference image detail hierarchy, "
                    "surface texture, and visual coherence."
                ),
                "loss of detail, over-smoothing, incorrect structure",
            ),
        ]

    return [
        (
            "network-control",
            (
                "Emphasize a connected neural decision network with cyan control nodes, "
                "structured signal flow, asymmetric composition, and a clear non-circular focal subject."
            ),
            "circular ring, concentric circles, target symbol, glowing ring, bullseye",
        ),
        (
            "molecular-glucose",
            (
                "Emphasize molecular glucose particles, distributed control signals, "
                "clean medical AI structure, and a premium futuristic scientific visual."
            ),
            "eyeball, iris, central eye, repeated ring pattern, portal",
        ),
        (
            "minimal-control",
            (
                "Emphasize a minimal premium AI control object with connected nodes, "
                "subtle holographic logic paths, and a visually clear subject without lens-like appearance."
            ),
            "camera lens, eye-like center, circular target, tunnel effect",
        ),
        (
            "data-field",
            (
                "Emphasize a reinforcement learning decision field with connected nodes, "
                "reward-signal trails, structured data flow, and reduced radial symmetry."
            ),
            "radial symmetry, portal, repeated circles, hypnotic circular texture",
        ),
        (
            "holographic-grid",
            (
                "Emphasize a clean holographic control grid with floating cyan nodes, "
                "glucose-related particles, and an elegant technical atmosphere."
            ),
            "bullseye composition, circular artifact, iris, eye, target pattern",
        ),
    ]


def build_prompt_variants(
    prompt: str,
    negative_prompt: str,
    evaluation_profile: str = "portfolio",
    num_variants: int = 4,
    avoid_radial_artifacts: bool = False,
    include_base: bool = True,
    portrait_reference_mode: bool = False,
) -> list[PromptVariant]:
    base_prompt = normalize_text(prompt)
    base_negative_prompt = normalize_text(negative_prompt)

    if avoid_radial_artifacts:
        base_negative_prompt = merge_negative_prompts(base_negative_prompt, ANTI_RADIAL_NEGATIVE)

    variants: list[PromptVariant] = []

    if include_base:
        variants.append(
            PromptVariant(
                label="base",
                prompt=base_prompt,
                negative_prompt=base_negative_prompt,
            )
        )

    for label, suffix, extra_negative in suffixes_for_profile(
        evaluation_profile=evaluation_profile,
        portrait_reference_mode=portrait_reference_mode,
    ):
        mutated_prompt = normalize_text(f"{base_prompt} {suffix}")
        mutated_negative_prompt = merge_negative_prompts(base_negative_prompt, extra_negative)

        variants.append(
            PromptVariant(
                label=label,
                prompt=mutated_prompt,
                negative_prompt=mutated_negative_prompt,
            )
        )

    if num_variants <= 0:
        return variants

    return variants[: max(1, num_variants)]