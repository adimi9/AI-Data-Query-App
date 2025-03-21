import re  

async def check_for_malicious_input(user_input: str):
    """
    This function checks if the user's input contains potentially malicious content,
    including SQL injection attempts, script injections, or other harmful keywords.

    Parameters:
    - user_input: str - The input string provided by the user.

    Returns:
    - A string message if malicious content is detected.
    - None if no malicious content is found.
    """
    
    # Check if the input is a string, if not, consider it potentially malicious
    if not isinstance(user_input, str):
        return "Your input appears to contain potentially malicious content and has been blocked."

    # List of potentially harmful keywords (SQL injection, XSS, etc.)
    malicious_keywords = [
        "drop table", 
        "insert into", 
        "delete from", 
        "exec ", 
        "javascript:", 
        "<script>", 
        "<iframe>", 
        "eval(", 
        "system(", 
        "shell_exec(", 
        "phpinfo()", 
    ]
    
    # Check if any of the malicious keywords are present in the user's input
    for keyword in malicious_keywords:
        if keyword.lower() in user_input.lower():
            return "Your input appears to contain potentially malicious content and has been blocked."

    # List of regex patterns to detect potential SQL injection attempts
    sql_injection_patterns = [
        r"\b(select|update|insert|delete)\b.*?\b(from|where|and|or)\b",  # SQL injection pattern
        r"'.*?--",  # Comment-based SQL injection
        r";--",  # SQL injection delimiter
    ]
    
    # Check if the input matches any SQL injection patterns
    for pattern in sql_injection_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            return "Your input appears to contain potentially malicious content and has been blocked."

    # If no malicious content is found, return None
    return None