from typing import Any, Dict, Optional, List

class VTOPromptFactory:
    """
    Modular prompt generator for Virtual Try-On.
    Supports Florence-2 guided "Defensive Prompting" to prevent color bleed.
    """

    def __init__(self):
        self.default_quality = (
            "Photorealistic high-fidelity virtual try-on, premium fashion photography, "
            "sharp details, clean background, 8k resolution."
        )

    def build_dynamic_prompt(
        self,
        garment_descriptions: List[str],
        user_description: Optional[str] = None,
        target_categories: Optional[List[str]] = None,
        is_high_vibrancy: bool = False
    ) -> str:
        """
        Constructs a prompt that specifically locks original garments while substituting others.
        """
        # 1. Base Identity Instruction
        prompt = (
            f"{self.default_quality} "
            "Use image 1 as strict identity source for face, body shape, and pose. "
        )

        # 2. Substitution Logic
        target_hint = ", ".join(garment_descriptions)
        prompt += f"TRANSFER the {target_hint} from image 2 onto the person. "

        # 3. Defensive Preservation Logic (The fix for color bleed)
        if user_description:
            prompt += f"STRICTLY PRESERVE all other original clothing items from image 1, especially the {user_description}. "
            prompt += "Do not bleed colors or textures from image 2 into the original garments. "
        else:
            prompt += "Maintain the style and color of the original untargeted clothing from image 1. "

        # 4. Fabric Awareness
        prompt += "Ensure realistic fabric drape, seams, and shadowing where the items fit the body."

        return prompt

# Singleton instance
prompt_factory = VTOPromptFactory()

if __name__ == "__main__":
    # Isolated Test
    test_p = prompt_factory.build_dynamic_prompt(
        garment_descriptions=["mustard yellow joggers"],
        user_description="charcoal gray ruffled t-shirt",
        is_high_vibrancy=True
    )
    print("--- GENERATED PROMPT ---")
    print(test_p)
