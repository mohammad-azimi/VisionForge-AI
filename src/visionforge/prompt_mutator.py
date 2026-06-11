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
    items = []

    for part in f"{base_negative_prompt}, {extra_negative_prompt}".split(","):
        cleaned = part.strip()
        if cleaned and cleaned.lower() not in {item.lower() for item in items}:
            items.append(cleaned)

    return ", ".join(items)


def suffixes_for_profile(evaluation_profile: str) -> list[tuple[str, str, str]]:
    profile = (evaluation_profile or "portfolio").strip().lower()

    if profile == "portrait":
        return [
            (
                "portrait-clean",
                (
                    "Emphasize a clean high-quality human portrait with natural facial structure, "
                    "clear eyes, realistic skin detail, balanced lighting, and strong facial coherence."
                ),
                "distorted face, extra eyes, deformed mouth, bad anatomy, duplicated facial features",
            ),
            (
                "portrait-cinematic",
                (
                    "Emphasize a cinematic portrait with clean depth, realistic expression, "
                    "professional composition, and subtle background separation."
                ),
                "extra face, double face, surreal anatomy, malformed eyes, malformed nose",
            ),
            (
                "portrait-minimal",
                (
                    "Emphasize a premium minimal portrait with a clear main subject, "
                    "simple background, polished lighting, and realistic proportions."
                ),
                "crowded scene, busy background, surreal body, extra limbs",
            ),
        ]

    if profile == "reference_match":
        return [
            (
                "reference-structure",
                (
                    "Emphasize preservation of the reference image structure, composition, "
                    "main subject silhouette, and overall scene layout."
                ),
                "major composition change, different framing, unrelated subject",
            ),
            (
                "reference-colors",
                (
                    "Emphasize preservation of the reference image color mood, "
                    "lighting atmosphere, and visual identity."
                ),
                "different color palette, wrong lighting, unrelated aesthetic",
            ),
            (
                "reference-clean",
                (
                    "Emphasize a clean faithful reconstruction of the reference image "
                    "with strong visual similarity and reduced artifacts."
                ),
                "messy reconstruction, noisy image, deformed shapes",
            ),
        ]

    # default: portfolio / general
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

    for label, suffix, extra_negative in suffixes_for_profile(evaluation_profile):
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