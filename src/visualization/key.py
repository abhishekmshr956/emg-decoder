class Key:
    def __init__(self, layout, ascii_code=None, char=None, coordinates=None, index=None):
        if ascii_code is None and char is None and coordinates is None and index is None:
            raise ValueError('Must provide one of code, char, coordinates, or index.')

        self.layout = layout
        self.flat_layout = [item for sublist in layout for item in sublist]
        self.flat_ascii = [ord(char) for char in self.flat_layout]
        self.flat_ascii_sorted = sorted(self.flat_ascii)

        self.ascii_code = None
        self.char = None
        self.coordinates = None
        self.index = None  # This is the index of the key after sorting by ascii code, for use with network outputs

        if ascii_code is not None:
            self.ascii_code = ascii_code
            self.char = chr(ascii_code)
            self.index = self.flat_ascii_sorted.index(self.ascii_code)
            self.coordinates = self.find_coords(self.char)

        elif char is not None:
            self.char = char
            self.ascii_code = ord(char)
            self.index = self.flat_ascii_sorted.index(self.ascii_code)
            self.coordinates = self.find_coords(self.char)

        elif index is not None:
            self.index = index
            self.ascii_code = self.flat_ascii_sorted[index]
            self.char = chr(self.ascii_code)
            self.coordinates = self.find_coords(self.char)

        elif coordinates is not None:
            self.coordinates = coordinates
            self.char = self.layout[coordinates[0]][coordinates[1]]
            self.ascii_code = ord(self.char)
            self.index = self.flat_ascii_sorted.index(self.ascii_code)

        if self.coordinates is None:
            raise ValueError('Key not found in layout.')

    def find_coords(self, char):
        coords = None
        for i in range(len(self.layout)):
            for j in range(len(self.layout[i])):
                if self.layout[i][j] == char:
                    coords = (i, j)
        return coords


if __name__ == '__main__':
    layout = [['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
              ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
              ['z', 'x', 'c', 'v', 'b', 'n', 'm'],
              [' '],
              [chr(0)]]
    a = Key(layout, ascii_code=0)
    print(a.ascii_code)
    print(a.char)
    print(a.coordinates)
    print(a.index)
    print(a.flat_ascii_sorted)
