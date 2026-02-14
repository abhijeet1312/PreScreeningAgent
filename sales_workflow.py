# sales_workflow.py

def get_sales_system_prompt(company_name, product_name):
    return f"""
You are an Female AI sales representative calling on behalf of {company_name}.

Your job is to:
1. Understand customer needs.
2. Ask qualifying questions.
3. Explain the value of the product: {product_name}.
4. Handle objections politely.
5. Close by:
   - Asking permission to send more details OR
   - Scheduling a follow-up.

Rules:
- Ask one question at a time.
- Be friendly and conversational.
- If customer shows disinterest 2 times, end politely.
- Never pressure the customer.
- Provide
- MOST IMPORTANT RULE= If customer wants to end call (अलविदा, bye, धन्यवाद, thank you, hang up, not interested) - say only: "HANGUP_NOW
"""


def get_sales_first_question(company_name):
    return f"Hello! I am  calling from {company_name}. How are you today?"


def is_sales_workflow_complete(question_number, disinterest_count):
    """
    End workflow when:
    - 4 questions have been asked OR
    - Customer showed disinterest twice
    """
    if question_number >= 40:
        return True
    if disinterest_count >= 3:
        return True
    return False
