# -*- coding: utf-8 -*-

# 1. parsimonious -> stack overflow
# 2. ANTLR        -> overcomplicated, Java unhappiness
# 3. tatsu        -> incorrect backtracking (strange, eh? It's even on version 4.2)
# 4. hand-rolled :( :(

import re
import string
import copy
from functools import reduce

# From the Haskell 2010 Report:
#
#   type   →  btype [-> type]        (function type)
#
#   btype  →  [btype] atype          (type application)
#
#   atype  →  gtycon
#          |  tyvar
#          |  ( type1 , … , typek )  (tuple type, k ≥ 2)
#          |  [ type ]               (list type)
#          |  ( type )               (parenthesised constructor)
#
#   gtycon →  qtycon
#          |  ()                     (unit type)
#          |  []                     (list constructor)
#          |  (->)                   (function constructor)
#          |  (,{,})                 (tupling constructors)
#
#   qtycon  →  [modid .] tycon
#   tycon →  conid                   (type constructors)
#
#
#
#   context →  class
#           |  ( class1 , … , classn )            (n ≥ 0)
#   class   →  qtycls tyvar
#           |  qtycls ( tyvar atype1 … atypen )   (n ≥ 1)
#   qtycls  →  [ modid . ] tycls
#   tycls   →  conid
#   tyvar   →  varid

# tatsu attempt:
#
#   start = Contextual_type $ ;
#
#   Contextual_type = MaybeWs contexts+:{Context} root_type:Root_type ;
#
#   Root_type = MaybeWs type_app:Type_app Arrow arrow_to:Root_type
#             | MaybeWs type_app:Type_app ;
#   Arrow     = MaybeWs "->"
#             | MaybeWs "→" ;
#   Type_app  = [t1:Type_app Ws] t2:Atype ;
#   Atype     = name:Name
#             | tuple:Tuple_type
#             | list:List_type
#             | parened:Parened_con ;
#
#   Tuple_type  = "(" first:Root_type "," rest:",".{Root_type}+ ")" ;
#   List_type   = "[" root:Root_type "]" ;
#   Parened_con = "(" root:Root_type ")" ;
#
#   Context = (Klass | Klasses) ("=>" | "⇒") ;
#
#   Klasses              = "(" Klass {"," Klass} ")" ;
#   Klass                = Simple_klass | Klass_on_constructed ;
#   Simple_klass         = Name {Ws Name}+ ;
#   Klass_on_constructed = Name "(" Name {Ws Atype}+ ")" ;
#
#   MaybeWs = /\s*/ ;
#   Ws      = /\s+/ ;
#
#   # Written this way to exclude bare arrows.
#   Name = /[^\-\s\[\]\(\),][^\s\[\]\(\),]+|[^\s\[\]\(\),][^>\s\[\]\(\),]+|\(\)|\[\]|\(->\)|\(,+\)|[^\s\[\]\(\),]/ ;
#

##### PARSING HELPERS ########

WS       = re.compile(r'\s+')

def parse_zero_or_more(parser, s):
  results = [([], s)]
  for result,s in parser(s):
    # results.append(([result],s))
    for deeper_results,s in parse_zero_or_more(parser,s):
      results.append(([result] + deeper_results,s))

  return results

def parse_n_or_more(parser, n, s):
  results = []
  for l,s in parse_zero_or_more(parser,s):
    if len(l) >= n:
      results.append((l,s))

  return results

def make_sep_ws_before_parser(sep,parser):
  def new_parser(s):
    results = []
    for _,s in parse_ws_str(sep,s):
      results += parser(s)
    return results

  return new_parser

def parse_n_or_more_separated_ws_by(parser, sep, n, s):
  if n == 0:
    results = [([],s)]
  else:
    results = []

  sep_and_parser = make_sep_ws_before_parser(sep,parser)

  for result,s in parser(s):
    for deeper_results,s in parse_n_or_more(sep_and_parser,n-1,s):
      results.append(([result] + deeper_results,s))

  return results

def parse_n_or_more_preceded_by_sep(parser, sep, n, s):
  if n == 0:
    return [([],s)]
  elif n < 0:
    return []

  results = []
  for _,s in parse_str(sep,s):
    for result,s in parser(s):
      for deeper_results,s in parse_n_or_more_preceded_by_sep(parser,sep,n-1,s):
        results.append(([result] + deeper_results,s))

  return results

def parse_ws_re(regex,s):
  s = s.lstrip()
  return parse_re(regex,s)

def parse_re(regex,s):
  match = re.match(regex,s) # Only matches at starts of strings.
  if match:
    return [(match.group(0),s[match.end():])]
  else:
    return []

def parse_ws_str(token,s):
  s = s.lstrip()
  return parse_str(token,s)

def parse_str(token,s):
  token_len = len(token)
  if s[0:token_len] == token:
    return [(token,s[token_len:])]
  else:
    return []

def require_ws(s):
  return parse_re(WS,s)


##### TYPE PARSING ######

WS_ARROW = re.compile(r' ->| →')
ARROW    = re.compile(r'->|→')
IMPLIES  = re.compile(r'=>|⇒')
# Written this way to exclude bare arrows and bare implies
# Includes: () [] (->) (,+)
NAME    = re.compile(r'[^\=\-\s\[\]\(\),][^\s\[\]\(\),]+|[^\s\[\]\(\),][^>\s\[\]\(\),]+|\(\)|\[\]|\(->\)|\(,+\)|[^\s\[\]\(\),→⇒]')

# Contextual_type = contexts+:{Context} root_type:Root_type ;
def parse_contextual_type(s):
  results = []
  for contexts,s in parse_zero_or_more(parse_context,s):
    for root,s in parse_root_type(s):
      results.append(({"contexts": contexts, "root_type": root}, s))

  return results

def parens_matched_no_top_level_commas(s):
  stack = []
  for char in s:
    if char == "(":
      stack.append(char)
    elif char == ")":
      if stack == [] or stack.pop() != "(":
        return False
    elif char == "[":
      stack.append(char)
    elif char == "]":
      if stack == [] or stack.pop() != "[":
        return False
    elif char == "," and stack == []:
      return False

  return len(stack) == 0

def first_index_at_same_level_ws(regex,s):
  for match in re.finditer(regex,s):
    if parens_matched_no_top_level_commas(s[0:match.start()]):
      return match.start()
  return None

# Root_type = MaybeWs type_app:Type_app Arrow arrow_to:Root_type
#           | MaybeWs type_app:Type_app ;
def parse_root_type(s):
  results = []
  s = s.lstrip()
  s_orig = s
  first_arrow_i = first_index_at_same_level_ws(WS_ARROW,s)
  if first_arrow_i != None:
    for app,s in parse_type_app(s[0:first_arrow_i]):
      # Nothing else parses bare arrows, so no partial match if there's an arrow.
      # results.append(({"type_app": app, "arrow_to": None},s))
      for _,s in parse_ws_re(ARROW,s_orig[first_arrow_i - len(s):]):
        for root,s in parse_root_type(s):
          results.append(({"type_app": app, "arrow_to": root},s))
  else:
    for app,s in parse_type_app(s):
      results.append(({"type_app": app, "arrow_to": None},s))

  return results

# Type_app  = [t1:Type_app Ws] t2:Atype ;
def parse_type_app(s):
  results = []
  orig_s = s
  for atype,s in parse_atype(orig_s):
    results.append(({"t1": None, "t2": atype},s))
    for args,s in parse_n_or_more(parse_require_ws_atype,1,s):
      result = reduce(lambda f,arg: {"t1":f, "t2":arg},[atype] + args,None)
      results.append((result,s))

  return results

def parse_require_ws_atype(s):
  results = []
  for _,s in require_ws(s):
    results += parse_atype(s)

  return results

# Atype = name:Name
#       | tuple:Tuple_type
#       | list:List_type
#       | parened:Parened_con ;
def parse_atype(s):
  results = []
  orig_s = s

  for name,s in parse_re(NAME,orig_s):
    results.append(({"name": name, "tuple": None, "list": None, "parened": None},s))
  if results != []:
    return results

  for tup,s in parse_tuple(orig_s):
    results.append(({"name": None, "tuple": tup, "list": None, "parened": None},s))
  if results != []:
    return results

  for l,s in parse_list(orig_s):
    results.append(({"name": None, "tuple": None, "list": l, "parened": None},s))
  if results != []:
    return results

  for parened,s in parse_parened_con(orig_s):
    results.append(({"name": None, "tuple": None, "list": None, "parened": parened},s))

  return results

def parse_ws_atype(s):
  s = s.lstrip()
  return parse_atype(s)


# Tuple_type  = "(" first:Root_type "," rest:",".{Root_type}+ MaybeWs ")" ;
def parse_tuple(s):
  results = []
  for _,s in parse_str("(",s):
    for elems,s in parse_n_or_more_separated_ws_by(parse_root_type,",",2,s):
      for _,s in parse_ws_str(")",s):
        results.append((elems,s))

  return results

# List_type   = "[" root:Root_type MaybeWs "]" ;
def parse_list(s):
  results = []
  for _,s in parse_str("[",s):
    for root,s in parse_root_type(s):
      for _,s in parse_ws_str("]",s):
        results.append((root,s))

  return results

# Parened_con = "(" root:Root_type MaybeWs ")" ;
def parse_parened_con(s):
  results = []
  for _,s in parse_str("(",s):
    for root,s in parse_root_type(s):
      for _,s in parse_ws_str(")",s):
        results.append((root,s))

  return results


# Context = (Klass | Klasses) MaybeWs ("=>" | "⇒") ;
def parse_context(s):
  results = []
  orig_s = s

  for klass,s in parse_klass(s):
    for _,s in parse_ws_re(IMPLIES,s):
      # Ignoring contexts for now, return unit.
      results.append(((),s))
  if results != []:
    return results

  for klasses,s in parse_klasses(s):
    for _,s in parse_ws_re(IMPLIES,s):
      # Ignoring contexts for now, return unit.
      results.append(((),s))

  return results

# Klasses = MaybeWs "(" Klass {"," Klass} MaybeWs ")" ;
def parse_klasses(s):
  results = []
  for _,s in parse_ws_str("(",s):
    for klasses,s in parse_n_or_more_separated_ws_by(parse_klass,",",1,s):
      for _,s in parse_ws_str(")",s):
        # Ignoring contexts for now, return unit.
        results.append(((),s))

  return results

# Klass = MaybeWs (Simple_klass | Klass_on_constructed) ;
def parse_klass(s):
  results = []
  s = s.lstrip()
  orig_s = s
  for simple_class,s in parse_simple_klass(orig_s):
    # Ignoring contexts for now, return unit.
    results.append(((),s))

  for klass_on_constructed,s in parse_klass_on_constructed(orig_s):
    # Ignoring contexts for now, return unit.
    results.append(((),s))

  return results


def parse_require_ws_name(s):
  results = []
  for _,s in require_ws(s):
    for name,s in parse_re(NAME,s):
      # Ignoring contexts for now, return unit.
      results.append(((),s))

  return results

# Simple_klass         = Name {Ws Name}+ ;
def parse_simple_klass(s):
  results = []
  for lname,s in parse_re(NAME,s):
    for _,s in parse_n_or_more(parse_require_ws_name,1,s):
      # Ignoring contexts for now, return unit.
      results.append(((),s))

  return results

# Klass_on_constructed = Name MaybeWs "(" MaybeWs Name {Ws Atype}+ MaybeWs ")" ;
def parse_klass_on_constructed(s):
  results = []
  for constraint_name,s in parse_re(NAME,s):
    for _,s in parse_ws_str("(",s):
      for ctor_name,s in parse_ws_re(NAME,s):
        for args,s in parse_n_or_more(parse_ws_atype,1,s):
          for _,s in parse_ws_str(")",s):
            # Ignoring contexts for now, return unit.
            results.append(((),s))

  return results


def parse_sig(sig_str):
  sig_str = sig_str.strip()
  full_results = []
  longest_parse = 0
  for result,s in parse_contextual_type(sig_str):
    if s == "":
      full_results.append(result)
    longest_parse = max(longest_parse, len(sig_str) - len(s))

  if len(full_results) == 1:
    return full_results[0]
  elif len(full_results) > 1:
    print(sig_str)
    for result in full_results:
      print(result)
    raise ValueError("ambiguous parse")
  elif len(full_results) == 0:
    print("could not parse:",sig_str)
    print("longest parse: '%s'" % sig_str[0:longest_parse])
    raise ValueError("incomplete parse")


# start = Contextual_type $ ;
#
# Contextual_type = MaybeWs contexts+:{Context} root_type:Root_type ;
#
# Root_type = MaybeWs type_app:Type_app Arrow arrow_to:Root_type
#           | MaybeWs type_app:Type_app ;
# Arrow     = MaybeWs "->"
#           | MaybeWs "→" ;
# Type_app  = [t1:Type_app Ws] t2:Atype ;
# Atype     = name:Name
#           | tuple:Tuple_type
#           | list:List_type
#           | parened:Parened_con ;
#
# Tuple_type  = "(" first:Root_type "," rest:",".{Root_type}+ ")" ;
# List_type   = "[" root:Root_type "]" ;
# Parened_con = "(" root:Root_type ")" ;
#
# Context = (Klass | Klasses) ("=>" | "⇒") ;
#
# Klasses              = "(" Klass {"," Klass} ")" ;
# Klass                = Simple_klass | Klass_on_constructed ;
# Simple_klass         = Name {Ws Name}+ ;
# Klass_on_constructed = Name "(" Name {Ws Atype}+ ")" ;
#
# MaybeWs = /\s*/ ;
# Ws      = /\s+/ ;
#
# # Written this way to exclude bare arrows.
# Name = /[^\-\s\[\]\(\),][^\s\[\]\(\),]+|[^\s\[\]\(\),][^>\s\[\]\(\),]+|\(\)|\[\]|\(->\)|\(,+\)|[^\s\[\]\(\),]/ ;


##### UNPARSING ############

UNPARSE_ARROW = "->"

def unparse_sig(sig):
  return unparse_root_type(sig["root_type"])

def unparse_root_type(root):
  left = unparse_type_app(root["type_app"])
  if root["arrow_to"]:
    return left + " " + UNPARSE_ARROW + " " + unparse_root_type(root["arrow_to"])
  else:
    return left

def unparse_type_app(app):
  if app["t1"]:
    return unparse_type_app(app["t1"]) + " " + unparse_atype(app["t2"])
  else:
    return unparse_atype(app["t2"])

def unparse_atype(atype):
  if atype["name"]:
    return atype["name"]
  elif atype["tuple"]:
    return "( " + " , ".join([unparse_root_type(elem) for elem in atype["tuple"]]) + " )"
  elif atype["list"]:
    return "[ " + unparse_root_type(atype["list"]) + " ]"
  elif atype["parened"]:
    return "( " + unparse_root_type(atype["parened"]) + " )"
  else:
    raise ValueError("bad atype in unparse: " + str(atype))

def normalize_type_variables(sig_str):
  sig = parse_sig(sig_str)
  renamings = {}
  normalize_sexp_type_variables_rec(renamings, sig_to_sexp(sig))
  return " ".join([renamings.get(token,token) for token in unparse_sig(sig).split()])

STRUCTURE = list("()[],") + [UNPARSE_ARROW]
def structure_only(sig_str):
  gathered = []
  for part in unparse_sig(parse_sig(sig_str)).split():
    if len(gathered) == 0 or gathered[-1] in STRUCTURE or part in STRUCTURE:
      gathered.append(part)
    else:
      gathered[-1] += " " + part

  renamings = {}
  out_parts = []
  for part in gathered:
    if part not in STRUCTURE and part not in renamings:
      renamings[part] = TYPE_VARS[len(renamings)].upper()

    out_parts.append(renamings.get(part,part))

  return " ".join(out_parts)



##### SEXP CONVERSION ######

SEXP_ARROW = UNPARSE_ARROW

# sexp of single applications
def sig_to_sexp(sig):
  if "root_type" in sig:
    return root_type_to_sexp(sig["root_type"])
  else:
    raise ValueError("could convert sig to sexp: " + str(sig))

def root_type_to_sexp(root):
  left = type_app_to_sexp(root["type_app"])
  if root["arrow_to"]:
    right = root_type_to_sexp(root["arrow_to"])
    return [[SEXP_ARROW, left], right]
  else:
    return left

def type_app_to_sexp(app):
  right = atype_to_sexp(app["t2"])
  if app["t1"]:
    left = type_app_to_sexp(app["t1"])
    if left == "->" or left == "(->)" or left == "(→)" or left == "→":
      left = SEXP_ARROW
    return [left, right]
  else:
    return right

# {'parened': None, 'list': None, 'name': 'a', 'tuple': None}
def atype_to_sexp(atype):
  if atype["name"]:
    return atype["name"]
  elif atype["tuple"]:
    count = len(atype["tuple"])
    ctor  = "(" + ","*(count-1) + ")"
    parts = [root_type_to_sexp(elem) for elem in atype["tuple"]]
    return reduce(lambda x,y: [x,y], parts, ctor)
  elif atype["list"]:
    return ["[]", root_type_to_sexp(atype["list"])]
  elif atype["parened"]:
    return root_type_to_sexp(atype["parened"])
  else:
    raise ValueError("bad atype in conversion to sexp: " + str(atype))


##### SEXP NORMALIZATION ######

LOWERCASE = re.compile(r'[a-z]')
# We elsewhere limit the size of type signatures we'll parse, so this is more than sufficient.
TYPE_VARS = list(string.ascii_lowercase) + [l1 + l2 for l1 in string.ascii_lowercase for l2 in string.ascii_lowercase]

def normalize_sexp_type_variables(sexp):
  return normalize_sexp_type_variables_rec({}, sexp)

def normalize_sexp_type_variables_rec(renamings, sexp):
  if type(sexp) is list:
    # Each sexp list should only have two elements, but we'll do this:
    return [normalize_sexp_type_variables_rec(renamings, elem) for elem in sexp]
  else:
    if LOWERCASE.match(sexp): # Only matches at starts of strings.
      if sexp not in renamings:
        renamings[sexp] = TYPE_VARS[len(renamings)]

      return renamings[sexp]
    else:
      return sexp

def sig_to_normalized_sexp(sig):
  return normalize_sexp_type_variables(sig_to_sexp(sig))

def sigs_equivalent(sig1, sig2):
  return sig_to_normalized_sexp(sig1) == sig_to_normalized_sexp(sig2)


#################################################
# Bowen's modification below

def flatten(l):
    '''
    flatten a tree in list representation
    '''
    if not l:
        return l
    if type(l[0]) is not list:
        return l
    return flatten(l[0]) + [l[1]]

class Tree():
    '''
    explicit tree representation of type signatures
    '''
    def __init__(self, normalized_sig):
        '''
        normalized_sig is the normalized type signature
        '''
        if type(normalized_sig) is str or type(normalized_sig) is int:
            self.node = normalized_sig
            self.left = None
            self.right = None
        else:
            assert(type(normalized_sig) is list)
            left = normalized_sig[0]
            right = normalized_sig[1]
            if left[0] == "->":
                self.node = "->"
                self.left = Tree(left[1])
                self.right = Tree(right)
            else:
                self.node = [Tree(x) if type(x) is list else x for x in flatten(normalized_sig)]
                self.left = None
                self.right = None

    @classmethod
    def from_str(cls, sig):
        normalized_sig = sig_to_normalized_sexp(parse_sig(sig))
        return Tree(normalized_sig)

    @classmethod
    def singleton(cls, node):
        res = Tree("")
        if type(node) is list and len(node) == 1:
          res.node = node[0]
        else:
          res.node = node
        return res

    @classmethod
    def from_children(cls, node, left_child, right_child):
        '''
        this method feels hacky
        it's supposed be wrap the results returned by the model
        '''
        assert(isinstance(left_child, Tree))
        assert(isinstance(right_child, Tree))
        res = Tree("")
        res.left = left_child
        res.right = right_child
        res.node = node
        return res

    def print_tree(self):
        '''
        print tree in dfs order. For debugging purposes
        '''
        print(self.node)
        if self.left is not None:
            self.left.print_tree()
        if self.right is not None:
            self.right.print_tree()

    def to_sig(self):
        if self.left is None and self.right is None:
            if type(self.node) is list:
                res = ""
                for t in self.node:
                    if type(t) is str:
                        res += t + " "
                    else:
                        res += "( " + t.to_sig() + " ) "
                return res.rstrip()
            else:
                return self.node
        else:
            return self.left.to_sig() + " -> " + self.right.to_sig()

    def decorate(self):
        '''
        decorate the nodes of the raw sig tree with their kinds
        '''
        if type(self.node) is str and self.node != "->":
            self.node += "#0"
        elif type(self.node) is list:
            self.node[0] = self.node[0] + "#" + str(len(self.node)-1)
            for i in range(1, len(self.node)):
                if type(self.node[i]) is str:
                    self.node[i] += "#0"
                else:
                    assert(isinstance(self.node[i], Tree))
                    self.node[i].decorate()
        if self.left is not None:
            self.left.decorate()
        if self.right is not None:
            self.right.decorate()

    def apply(self, func):
        if type(self.node) is str or type(self.node) is int:
            self.node = func(self.node)
        else:
            assert(type(self.node) is list)
            self.node = [x.apply(func) if isinstance(x, Tree) else func(x) for x in self.node]
        if self.left is not None:
            self.left.apply(func)
        if self.right is not None:
            self.right.apply(func)
        return self

    def to_index(self, dict, unk_token, oov_dict=None):
        self.decorate()
        def func(x):
            if x[0].isalpha():
                x = x.split('.')[-1]
            if dict.get(x) is not None:
                return dict[x]
            elif oov_dict is not None and oov_dict.get(x) is not None:
                return oov_dict[x]
            return unk_token
        return self.apply(func)

    def to_index_augment(self, dict, unk_token,
                         oov_token_to_idx, oov_idx_to_token, oov_kind_dict):
        self.decorate()
        def func(x):
            if x[0].isalpha():
                x = x.split('.')[-1]
            if dict.get(x) is None:
                cur_num = len(dict) + len(oov_token_to_idx)
                oov_token_to_idx[x] = cur_num
                oov_idx_to_token[cur_num] = x
                kind = int(x.rsplit('#', 1)[-1])
                oov_kind_dict[cur_num] = kind
                return unk_token
            return dict.get(x)
        return self.apply(func)

    def to_str(self, dict, oov_dict=None):
        def func(x):
            if dict.get(x) is None:
                assert(oov_dict is not None)
                return oov_dict[x].rsplit('#', 1)[0]
            return dict[x].rsplit('#', 1)[0]
        return self.apply(func)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.left is None:
                left_flag = other.left is None
            else:
                left_flag = self.left.__eq__(other.left)
            if self.right is None:
                right_flag = other.right is None
            else:
                right_flag = self.right.__eq__(other.right)
            return left_flag and right_flag and self.node == other.node
        return False

    def node_count(self):
        count = 0
        if type(self.node) is str or type(self.node) is int:
            count += 1
        else:
            assert(type(self.node) is list)
            for item in self.node:
                if isinstance(item, self.__class__):
                    count += item.node_count()
                else:
                    count += 1
        if self.left is not None:
            count += self.left.node_count()
        if self.right is not None:
            count += self.right.node_count()
        return count

    def traversal(self, dict=None, ignore_node=None):
        node_map = {} if dict is None else dict
        if type(self.node) is str or type(self.node) is int:
            if self.node != ignore_node:
                node_map[len(node_map)] = self.node
        else:
            assert(type(self.node) is list)
            for item in self.node:
                if isinstance(item, self.__class__):
                    item.traversal(dict=node_map, ignore_node=ignore_node)
                elif item != ignore_node:
                    node_map[len(node_map)] = item
        if self.left is not None:
            self.left.traversal(dict=node_map, ignore_node=ignore_node)
        if self.right is not None:
            self.right.traversal(dict=node_map, ignore_node=ignore_node)
        return node_map

    def get_last(self):
        if self.right is not None:
            return self.right.get_last()
        if type(self.node) is list:
            if isinstance(self.node[-1], self.__class__):
                return self.node[-1].get_last()
            else:
                return self.node[-1]
        return self.node

    def strip(self):
        '''
        a helper function for structural comparison
        '''
        self.node = ""
        if self.left is not None:
            self.left.strip()
        if self.right is not None:
            self.right.strip()

    def structural_eq(self, other):
        if isinstance(other, self.__class__):
            self_copy = copy.deepcopy(self)
            other_copy = copy.deepcopy(other)
            self_copy.strip()
            other_copy.strip()
            return self_copy == other_copy
        return False

    def depth(self):
      left_depth = 0
      right_depth = 0
      if self.left is not None:
        left_depth = self.left.depth()
      if self.right is not None:
        right_depth = self.right.depth()
      cur_depth = 1
      if type(self.node) is list:
        for item in self.node:
          if isinstance(item, self.__class__):
            cur_depth = max(cur_depth, item.depth())
      return cur_depth + max(left_depth, right_depth)


'''
sig = "Maybe M.Int"

print sig
print parse_sig(sig)
print unparse_sig(parse_sig(sig))
print sig_to_sexp(parse_sig(sig))
print sig_to_normalized_sexp(parse_sig(sig))
print(sig_to_normalized_sexp(parse_sig("Maybe (Int -> Int) -> [Int] -> Either Int String")))

tree = Tree.from_str("Maybe (IO a) -> [(Int, Int)] -> Either Int String")
print(tree.to_sig())
print(tree.traversal(ignore_node="->"))
test_dict = {"->": 1, "Maybe#1": 2, "IO#1": 3, "a#0": 4, "[]#1": 5, "Int#0": 6, "Either#2": 7, "String#0": 8}
test_dict1 = {1: "->", 2: "Maybe#1", 3: "IO#1", 4: "a#0", 5: "[]#1", 6: "Int#0", 7: "Either#2", 8: "String#0"}
tree1 = Tree.from_str("Maybe (IO a) -> [Int] -> Either Int String")
tree1.to_index(test_dict, 0)
print(tree1.to_str(test_dict1).to_sig())
'''
