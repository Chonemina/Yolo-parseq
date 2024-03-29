#chua cac ham co the hoan thanh tac vu chinh su nhung class con thieu
#ham tim xem quest_id nao con thieu
Example=["câu 51","câu 52","câu 54"]
def lostquest_id(input_list):
    
    input_string= ",".join(input_list)
# Tách chuỗi thành danh sách các phần tử
    elements = input_string.split(",")

# Tạo danh sách các số nguyên tương ứng
    int_elements = [int("".join(filter(str.isdigit, ele))) for ele in elements]
    
    missing_number=[]
    for i in range(min(int_elements),max(int_elements)):
        if i not in int_elements:
            missing_number.append(i)
    if not missing_number:
        print ("Khong con thieu cau hoi nao")
    else:
        print("Các quest_id còn thiếu:", missing_number)
 
#lostquest_id(Example)