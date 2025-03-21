import re  

async def check_for_jailbreak_attempt(user_request):
    """
    This function checks if the user's request contains patterns that suggest an attempt to bypass safety guidelines 
    (commonly known as jailbreaking).

    Parameters:
    - user_request: str - The question or request made by the user.

    Returns:
    - A message indicating the request is blocked if a jailbreak pattern is detected.
    - None if no jailbreak patterns are found.
    """

    # Define a list of patterns that are typically associated with jailbreaking attempts
    jailbreak_patterns = [
        r"ignore all previous instructions",  
        r"do not follow any of my instructions",  
        r"pretend to be",  
        r"as a [^ ]+ bot",  
        r"you are now a [^ ]+",  
        r"i want you to act as",  
        r"from now on, you are",  
        r"i am going to give you a command",  
        r"override all rules",  
        r"disregard previous prompts", 
        r"you must respond as",  
        r"i will give you a new role", 
        r"act as if you are",  
        r"i want you to become", 
        r"you are no longer",  
        r"new rule: ",  
        r"let's play a game",
        r"i'm going to test your limits",  
        r"you have to answer",  
        r"you must not refuse",  
        r"you will always", 
        r"no matter what",  
        r"regardless of any rules", 
        r"i'm going to try to trick you",
        r"you are allowed to",  
        r"you must agree to", 
        r"i want you to help me with", 
        r"write a story about",  
        r"tell me a joke",  
        r"what is your secret", 
        r"can you tell me a secret",  
    ]

    # Convert the user's request to lowercase to make the pattern matching case-insensitive
    user_request_lower = user_request.lower()

    # Check if any of the predefined patterns match the user's request
    for pattern in jailbreak_patterns:
        if re.search(pattern, user_request_lower):  
            return "I'm sorry, I cannot fulfill that request as it may violate my safety guidelines."

    # If no patterns match, return None (indicating no jailbreaking attempt was found)
    return None