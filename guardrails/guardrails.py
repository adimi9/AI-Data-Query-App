# Import necessary guardrail functions for input validation
from .topical_guardrail import check_for_irrelevant_input
from .jailbreak_guardrail import check_for_jailbreak_attempt
from .malicious_input_guardrail import check_for_malicious_input

async def execute_input_guardrails(user_request, sample_df):
    """
    Executes all the input guardrails to validate user input.

    Args:
        user_request: str - The user's input string that needs to be validated.
        sample_df: DataFrame - A sample DataFrame for data-related checks.

    Returns:
        str: A message if any guardrail is triggered, otherwise None if all checks pass.
    """

    # --- Jailbreaking Guardrail ---
    # Check if the input contains any jailbreaking attempts (e.g., bypassing safety rules)
    jailbreak_result = await check_for_jailbreak_attempt(user_request)
    if jailbreak_result:
        return jailbreak_result  

    # --- Topical Guardrail ---
    # Check if the input is relevant to the uploaded dataset (e.g., if it refers to the dataset columns)
    topical_result = await check_for_irrelevant_input(user_request, sample_df)
    if topical_result:
        return topical_result  
    
    # --- Malicious Input Guardrail ---
    # Check if the input contains any malicious content, such as SQL injection or script injections
    malicious_result = await check_for_malicious_input(user_request)
    if malicious_result:
        return malicious_result  

    # No guardrail was triggered, meaning the input is safe and relevant
    return None
