import unittest

from ai.modules.vto.prompt_factory import prompt_factory


class TestPromptFactory(unittest.TestCase):
    def test_prompt_includes_preservation_clause(self):
        prompt = prompt_factory.build_dynamic_prompt(
            garment_descriptions=["mustard yellow joggers"],
            user_description="charcoal gray t-shirt",
        )
        self.assertIn("STRICTLY PRESERVE", prompt)
        self.assertIn("mustard yellow joggers", prompt)

    def test_prompt_without_user_description(self):
        prompt = prompt_factory.build_dynamic_prompt(
            garment_descriptions=["black blazer"],
            user_description=None,
        )
        self.assertIn("Maintain the style and color", prompt)


if __name__ == "__main__":
    unittest.main()

