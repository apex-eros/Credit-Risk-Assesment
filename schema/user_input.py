from pydantic import BaseModel

class CustomerData(BaseModel):
    Age: int
    Sex: int
    Job: int
    Saving_accounts: int
    Checking_account: int
    Credit_amount: float
    Duration: int
    Housing_own: int
    Housing_rent: int
    Purpose : int
