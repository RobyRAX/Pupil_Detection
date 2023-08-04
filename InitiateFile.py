import openpyxl

def create_new_sheet(workbook, filename, sheet_name):

    # Create a new sheet with the specified name
    workbook.create_sheet(title=sheet_name)

    # Save the workbook to the specified filename
    workbook.save(filename)

if __name__ == "__main__":
    filename = "Hasil/Hasil.xlsx"  # Replace with the desired filename
    sheet_1 = "Normal"
    sheet_2 = "Autis"  # Replace with the desired sheet name
    
    # Create a new Excel workbook
    workbook = openpyxl.Workbook()
    # Remove the default sheet created by openpyxl (Sheet)
    default_sheet = workbook.active
    workbook.remove(default_sheet)

    create_new_sheet(workbook, filename, sheet_1)
    create_new_sheet(workbook, filename, sheet_2)
