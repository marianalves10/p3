class uCType:
    """
    Class that represents a type in the uC language.  Basic
    Types are declared as singleton instances of this type.
    """

    def __init__(
        self, name, binary_ops=set(), unary_ops=set(), rel_ops=set(), assign_ops=set()
    ):
        """
        You must implement yourself and figure out what to store.
        """
        self.typename = name
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.rel_ops = rel_ops
        self.assign_ops = assign_ops


# Create specific instances of basic types. You will need to add
# appropriate arguments depending on your definition of uCType
IntType = uCType(
    "int",
    unary_ops={"-", "+"},
    binary_ops={"+", "-", "*", "/", "%"},
    rel_ops={"==", "!=", "<", ">", "<=", ">="},
    assign_ops={"="},
)

CharType = uCType("char",
                  rel_ops={"==", "!=", "&&", "||"},
                  assign_ops={"="},

)
BoolType = uCType("bool",
                  unary_ops={"!"},
                  binary_ops={},
                  rel_ops={"==", "!=", "&&", "||"},
                  assign_ops={"="},
)

VoidType =  uCType("void"
)

StringType = uCType(
    "string",
    rel_ops={"==", "!="},
    assign_ops={"="}

)

# TODO: add array and function types
# Array and Function types need to be instantiated for each declaration
class ArrayType(uCType):
    def __init__(self, element_type, size=None):
        """
        type: Any of the uCTypes can be used as the array's type. This
              means that there's support for nested types, like matrices.
        size: Integer withas_returnh the length of the array.
        """
        self.type = element_type
        self.size = size
        super().__init__(None, rel_ops={"==", "!="})

        self.typename = "array"
class FunctionType(uCType):
    def __init__(self, binary_ops=set(), unary_ops=set(), rel_ops=set(), assign_ops=set(), return_type = None, param = []):
        self.return_type = return_type
        self.parameters = param

        super().__init__(None, binary_ops, unary_ops, rel_ops, assign_ops)
    def __str__(self):
        return f"{self.return_type}, {self.parameters}"
