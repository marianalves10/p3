import argparse
import pathlib
import sys
from copy import deepcopy
from typing import Any, Dict, Union
from uc.uc_ast import ID
from uc.uc_parser import UCParser
from uc.uc_type import CharType, IntType, VoidType, BoolType, ArrayType, FunctionType


class SymbolTable:
    """Class representing a symbol table.

    `add` and `lookup` methods are given, however you still need to find a way to
    deal with scopes.

    ## Attributes
    - :attr data: the content of the SymbolTable
    """

    def __init__(self) -> None:
        """ Initializes the SymbolTable. """
        self.scope = []

    def create_scope(self):
        self.scope.insert(0, {})
    def destroy_scope(self):
        if self.scope:
            self.scope.pop(0)
    def add(self, name: str, value: Any) -> None:
        """ Adds to the SymbolTable.

        ## Parameters
        - :param name: the identifier on the SymbolTable
        - :param value: the value to assign to the given `name`
        """
        self.scope[0][name] = value

    def lookup(self, name: str) -> Union[Any, None]:
        for scope in self.scope:
            res = scope.get(name)
            if res:
                return res
        return None


class NodeVisitor:
    """A base NodeVisitor class for visiting uc_ast nodes.
    Subclass it and define your own visit_XXX methods, where
    XXX is the class name you want to visit with these
    methods.
    """

    _method_cache = None

    def visit(self, node):
        """Visit a node."""

        if self._method_cache is None:
            self._method_cache = {}

        visitor = self._method_cache.get(node.__class__.__name__)
        if visitor is None:
            method = "visit_" + node.__class__.__name__
            visitor = getattr(self, method, self.generic_visit)
            self._method_cache[node.__class__.__name__] = visitor

        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a
        node. Implements preorder visiting of the node.
        """
        for _, child in node.children():
            self.visit(child)


class Visitor(NodeVisitor):
    """
    Program visitor class. This class uses the visitor pattern. You need to define methods
    of the form visit_NodeName() for each kind of AST node that you want to process.
    """

    def __init__(self):
        # Initialize the symbol table
        self.symtab = SymbolTable()
        self.typemap = {
            "int": IntType,
            "char": CharType,
            "void": VoidType,
            "bool": BoolType
        }
        self.has_return = False

    def _assert_semantic(self, condition: bool, msg_code: int, coord, name: str = "", ltype="", rtype=""):
        """Check condition, if false print selected error message and exit"""
        error_msgs = {
            1: f"{name} is not defined",
            2: f"subscript must be of type(int), not {ltype}",
            3: "Expression must be of type(bool)",
            4: f"Cannot assign {rtype} to {ltype}",
            5: f"Binary operator {name} does not have matching LHS/RHS types",
            6: f"Binary operator {name} is not supported by {ltype}",
            7: "Break statement must be inside a loop",
            8: "Array dimension mismatch",
            9: f"Size mismatch on {name} initialization",
            10: f"{name} initialization type mismatch",
            11: f"{name} initialization must be a single element",
            12: "Lists have different sizes",
            13: "List & variable have different sizes",
            14: f"conditional expression is {ltype}, not type(bool)",
            15: f"{name} is not a function",
            16: f"no. arguments to call {name} function mismatch",
            17: f"Type mismatch with parameter {name}",
            18: "The condition expression must be of type(bool)",
            19: "Expression must be a constant",
            20: "Expression is not of basic type",
            21: f"{name} does not reference a variable of basic type",
            22: f"{name} is not a variable",
            23: f"Return of {ltype} is incompatible with {rtype} function definition",
            24: f"Name {name} is already defined in this scope",
            25: f"Unary operator {name} is not supported",
        }
        if not condition:
            msg = error_msgs[msg_code]  # invalid msg_code raises Exception
            print("SemanticError: %s %s" % (msg, coord), file=sys.stdout)
            sys.exit(1)

    def visit_Program(self, node):
        # Visit all of the global declarations
        print("inicio programa")
        self.symtab.create_scope()
        for _decl in node.gdecls:
            self.visit(_decl)
        self.symtab.destroy_scope()
        # TODO: Manage the symbol table

    def visit_BinaryOp(self, node):
        # Visit the left and right expression
        print("binaryop")
        self.visit(node.lvalue)
        ltype = node.lvalue.uc_type
        self.visit(node.rvalue)
        rtype = node.rvalue.uc_type
        self._assert_semantic(rtype == ltype, 4,  ltype, rtype)
        self._assert_semantic(node.op in ltype.binary_ops.union(ltype.rel_ops), 6, node.coord, node.op, f"type({ltype.typename})")
        # TODO:
        # - Make sure left and right operands have the same type
        # - Make sure the operation is supported
        # - Assign the result type
        if node.op in ltype.rel_ops:
            node.uc_type = self.typemap["bool"]
        else:
            node.uc_type = ltype
    def visit_UnaryOp(self, node):
        print("unaryop")
        self.visit(node.expr)
        self._assert_semantic(node.op in node.expr.uc_type.unary_ops, 25, node.coord, node.op)
        node.uc_type = node.expr.uc_type
    def visit_DeclList(self, node):
        print("decllist")
        for decl in (node.decls or []):
            self.visit(decl)
    def visit_VarDecl(self, node):
        print("var decl")
        node.uc_type = self.visit(node.type)
    def visit_Assignment(self, node):
        # visit right side
        print("assigment")
        self.visit(node.rvalue)
        rtype = node.rvalue.uc_type
        # visit left side (must be a location)
        _var = node.lvalue
        self.visit(_var)
#         if isinstance(_var, ID):
#             self._assert_semantic(_var.scope is not None,
#                                   1, node.coord, name=_var.name)
        ltype = node.lvalue.uc_type
        # Check that assignment is allowed
        self._assert_semantic(ltype == rtype, 4, node.coord,
                              ltype=ltype, rtype=rtype)
        # Check that assign_ops is supported by the type
        self._assert_semantic(
            node.op in ltype.assign_ops, 5, node.coord, name=node.op, ltype=ltype
        )

    def visit_Constant(self, node):
        print("constant")
        node.uc_type = self.typemap[node.type]
        if node.type == "int":
            node.value = int(node.value)

    def visit_Type(self, node):
        print("type")
        node.uc_type = self.typemap[node.name]

    def visit_ID(self, node):
        print("ID")
        value_node = self.symtab.lookup(node.name)
        self._assert_semantic(value_node is not None, 1, node.coord, node.name)
        node.uc_type = value_node.type.uc_type

    def visit_Assert(self, node):
        print("assert")
        self.visit(node.expr)
        node.uc_type = node.expr.uc_type
        self._assert_semantic(node.uc_type == BoolType, 3)

    def visit_Compound(self, node):
        #VAI QUE É VAZIO
        print("compound")
        if node.citens:
            for i in node.citens:
                self.visit(i)

    def visit_If(self, node):
        print("if")
        self.visit(node.cond)
        self._assert_semantic(node.cond.uc_type == BoolType, 18, node.coord)
        self.visit(node.iftrue)

        if node.iffalse:
            self.visit(node.iffalse)

    def check_type(self, type):
        if type in [IntType, BoolType, CharType]:
            return True
        else:
            return False

    def visit_Decl(self, node):
        print("decl")

        self._assert_semantic(self.symtab.lookup(node.name) is None,
                              24, node.name.coord, name=node.name.name)

        self.symtab.add(node.name.name, node.type)

        self.visit(node.type)
        node.name.uc_type = node.type.type.uc_type
        node.uc_type = node.type.type.uc_type
        if node.init:
            self.visit(node.init)
            if not node.uc_type == ArrayType:
                self._assert_semantic(self.check_type(node.init.uc_type), 11, node.name.coord, name=node.name.name)


    def visit_FuncDef(self, node):
        print("funcdef")
        #print("node type", node.type)
        self.visit(node.type)
        self.visit(node.decl)
        #print()
        #print("node.type.type.params",node.type.type.params)
        if node.type.type.params is not None:
            params_type = [decl.uc_type for decl in node.type.type.params.params]
        else:
            params_type = []
        #print("lista", params_type)
        self.symtab.add(node.type.name, FunctionType(return_type=node.decl.uc_type, param= params_type))

        #self.symtab.add(node.type.type.params.params[0].name.name,IntType)
        self.has_return = False
        self.return_type = node.decl.name
        self.symtab.function_scope = True
        self.visit(node.body)
        self._assert_semantic(self.return_type == "void" or self.has_return, 23, node.body.coord, ltype=f'type({VoidType.typename})', rtype=f'type({self.return_type})')
        self.symtab.function_scope = False
        self.symtab.destroy_scope()
        self.return_type = None
    # vardecl: fazer um add na symtab,
    # definicao de funcao, precis ccriar um scope novo, fazer uma pilha de symtab.
    # quando define uma funcao, guarda o tipo do return (quando usar o return, tem que ver se o tipo ta correto)
    #  para o break, guardar que to dentro de uma funcao

    def visit_Return(self, node):
        self.has_return = True

    def visit_FuncDecl(self, node):
        print("funcdecl")
        self.symtab.create_scope()

        self.visit(node.type)

        if node.params:
            params = self.visit(node.params)
        else:
            params = []

        node.uc_type = FunctionType(node.type.uc_type, params)
if __name__ == "__main__":
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", help="Path to file to be semantically checked", type=str
    )
    args = parser.parse_args()

    # get input path
    input_file = args.input_file
    input_path = pathlib.Path(input_file)

    # check if file exists
    if not input_path.exists():
        print("Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    # set error function
    p = UCParser()
    # open file and parse it
    with open(input_path) as f:
        ast = p.parse(f.read())
        sema = Visitor()
        sema.visit(ast)
