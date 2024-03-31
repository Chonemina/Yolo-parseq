#chua cac ham co the hoan thanh tac vu chinh su nhung class con thieu
#ham tim xem quest_id nao con thieu
Example=["câu51.","câu52.","câu 55."]
"""
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
    return missing_number
#a=lostquest_id(Example)
#print(a)
"""
def is_inside(rect, rectlist):
    
    for r in rectlist:
        [x1,y1,x2,y2]=r
        if x1 <= rect[0] <= x2 and y1 <= rect[1] <= y2:
            return True
    return False
   
    

rect_list = [[0, 0, 5, 6], [2, 2, 3, 6], [10, 10, 44, 15]]
rect = [3, 3, 8, 4]  # Tọa độ của rect (góc trên bên trái: (3, 3), góc dưới bên phải: (4, 4))

#if is_inside(rect, rect_list):
 #   print("rect nằm hoàn toàn bên trong các hìn trong rect_list.")
#else:
  #  print("rect không nằm hoàn toàn bên trong các hìnhtrong rect_list.")