"""
Module 07 - Tools and Agents: 01_custom_tools.py
Defining typed tools with `@tool` and Pydantic input schemas.
"""
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class CurrencyConversionInput(BaseModel):
    amount: float = Field(description="The numeric currency amount to convert")
    from_currency: str = Field(description="3-letter source currency ISO code, e.g. USD, EUR, GBP")
    to_currency: str = Field(description="3-letter target currency ISO code, e.g. USD, EUR, JPY")

@tool(args_schema=CurrencyConversionInput)
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Converts an amount from one fiat currency to another using fixed benchmark rates."""
    rates_to_usd = {"USD": 1.0, "EUR": 1.08, "GBP": 1.28, "JPY": 0.0065}
    from_curr = from_currency.upper()
    to_curr = to_currency.upper()

    if from_curr not in rates_to_usd or to_curr not in rates_to_usd:
        return f"Error: Unsupported currency code. Supported: {list(rates_to_usd.keys())}"

    usd_amount = amount * rates_to_usd[from_curr]
    converted = usd_amount / rates_to_usd[to_curr]
    return f"{amount} {from_curr} = {converted:.2f} {to_curr}"

def main():
    print("=== Custom Tool Definition & Schema Inspection ===")
    print(f"Tool Name: {convert_currency.name}")
    print(f"Tool Description: {convert_currency.description}")
    print(f"Tool Arguments Schema:")
    print(convert_currency.args)

    print("\nExecuting tool directly:")
    result = convert_currency.invoke({"amount": 250, "from_currency": "EUR", "to_currency": "USD"})
    print(f"Execution Result: {result}")

if __name__ == "__main__":
    main()
