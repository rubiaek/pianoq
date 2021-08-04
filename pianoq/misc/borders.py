class Borders(object):
    def __init__(self, min_x, min_y, max_x, max_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

    @property
    def width(self):
        return self.max_x - self.min_x

    @property
    def height(self):
        return self.max_y - self.min_y

    @property
    def center_x(self):
        return (self.min_x + self.max_x) / 2

    @property
    def center_y(self):
        return (self.min_y + self.max_y) / 2

    def __repr__(self):
        return f'Borders(x: [{self.min_x}-{self.max_x}], y: [{self.min_y}-{self.max_y}])'
