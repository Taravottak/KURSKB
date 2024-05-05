# In code1.py
variable_to_extract = "vlue"

# In code2.py
def check_variable(variable):
    if variable == "value":
        print("Variable is valid")
    else:
        print("Variable is invalid")

# Call the function with the variable from code1.py
check_variable(variable_to_extract)
