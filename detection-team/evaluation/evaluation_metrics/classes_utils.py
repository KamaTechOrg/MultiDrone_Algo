class BoundingBox():
    def __init__(self, category_id, xmin, ymin, xmax, ymax) -> None:
        # print(category_id)
        self.__category_id=category_id
        self.__xmin=xmin
        self.__ymin=ymin
        self.__xmax=xmax
        self.__ymax=ymax
    
    def get_bbox(self):
        return (self.__xmin,self.__ymin,self.__xmax,self.__ymax)
    def get_category(self):
        return self.__category_id


class OutputObject():
    def __init__(self, image_path:str, bboxes:list[BoundingBox]):
        self.__image_path=image_path
        self.bboxes=bboxes