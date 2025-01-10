
```markdown
# Assembly Language Grammar

This document defines the grammar of an assembly language, covering program structure, instructions, directives, and expressions.

## Program Structure

```ebnf
program = { topLevelElement }, eof ;

topLevelElement = instruction [ comment ] lineEnd
| directive [ comment ] lineEnd
| macroDefinition
| templateDefinition
| moduleDefinition
| registerClassDefinition
| threadDefinition
| enumDefinition
| structDefinition
| threadLocalDirective [ comment ] lineEnd
| namespaceDefinition
| conceptDefinition
| procedureDefinition ;
```

## Namespaces

```ebnf
namespaceDefinition = "namespace", identifier, "{", { topLevelElement }, "}" ;
```

## Concepts

```ebnf
conceptDefinition = "concept", identifier, [ "<", templateParameterList, ">" ], [ whereClause ], "{", { conceptRequirement }, "}" ;
conceptRequirement = typeRequirement
| expressionRequirement ;
typeRequirement = "typename", identifier, ":", templateParameter ;
expressionRequirement = "requires", expression, ";" ;
whereClause = "where", expression ;
```

## Threads

```ebnf
threadDefinition = "thread", identifier, [ "<", templateArgumentList, ">" ], [ "(", parameterList, ")" ], [ ":", typeReference ], "{", { topLevelElement }, "}" ;
parameterList = parameter, { ",", parameter } ;
parameter = identifier, ":", typeReference ;
```

## Procedures/Functions

```ebnf
procedureDefinition = "proc", identifier, [ "(", parameterList, ")" ], [ "->", typeReference ], [ "callingconvention", identifier ], "{", { topLevelElement }, "}" ;
```

## Instructions

```ebnf
instruction = [ label, ":" ], [ instructionPrefix ], architectureSpecificInstructionBody ;

architectureSpecificInstructionBody = x86InstructionBody  (* @arch: x64, x86-32 *)
| armInstructionBody  (* @arch: arm *)
| gpuInstructionBody  (* @arch: gpu *) ;
```

### X86 Instructions

```ebnf
x86InstructionBody = [ x86InstructionPrefix ], x86Mnemonic, [ x86OperandList ]
| x86ShorthandInstruction ;
x86ShorthandInstruction = x86ModifiableOperand, shorthandOperator, expression ;
x86Mnemonic = [ namespaceQualifier ], x86InstructionMnemonic ;
x86InstructionMnemonic = "mov"  (* @arch: all x86 *)
| "add"  (* @arch: all x86 *)
| "sub"  (* @arch: all x86 *)
| "jmp"  (* @arch: all x86 *)
| "call" (* @arch: all x86 *)
| "ret"  (* @arch: all x86 *)
| "push" (* @arch: all x86 *)
| "pop"  (* @arch: all x86 *)
| "lea"  (* @arch: all x86 *)
| "cmp"  (* @arch: all x86 *)
| "test" (* @arch: all x86 *)
| "and"  (* @arch: all x86 *)
| "or"   (* @arch: all x86 *)
| "xor"  (* @arch: all x86 *)
| "not"  (* @arch: all x86 *)
| "neg"  (* @arch: all x86 *)
| "mul"  (* @arch: all x86 *)
| "imul" (* @arch: all x86 *)
| "div"  (* @arch: all x86 *)
| "idiv" (* @arch: all x86 *)
| "shl"  (* @arch: all x86 *)
| "shr"  (* @arch: all x86 *)
| "sar"  (* @arch: all x86 *)
| "rol"  (* @arch: all x86 *)
| "ror"  (* @arch: all x86 *)
| "rcl"  (* @arch: all x86 *)
| "rcr"  (* @arch: all x86 *)
| "jz"   (* @arch: all x86 *)
| "jnz"  (* @arch: all x86 *)
| "je"   (* @arch: all x86 *)
| "jne"  (* @arch: all x86 *)
| "js"   (* @arch: all x86 *)
| "jns"  (* @arch: all x86 *)
| "jg"   (* @arch: all x86 *)
| "jge"  (* @arch: all x86 *)
| "jl"   (* @arch: all x86 *)
| "jle"  (* @arch: all x86 *)
| "ja"   (* @arch: all x86 *)
| "jae"  (* @arch: all x86 *)
| "jb"   (* @arch: all x86 *)
| "jbe"  (* @arch: all x86 *)
;
x86InstructionPrefix = { x86RepeatPrefix }, { x86SegmentPrefix }, { x86AddressPrefix }, { x86DataPrefix }, { x86VectorPrefix }, { x86OtherPrefix } ;
x86RepeatPrefix = "rep" | "repe" | "repz" | "repne" | "repnz" | "lock" ;
x86SegmentPrefix = "cs" | "ds" | "es" | "fs" | "gs" | "ss" ;
x86AddressPrefix = "addr16" | "addr32" | "addr64" ;
x86DataPrefix = "byte" | "word" | "dword" | "qword" | "tbyte" ;
x86VectorPrefix = "xmmword" | "ymmword" | "zmmword" ;
x86OtherPrefix = "bnd" | "notrack" | "gfx" ;
x86OperandList = x86Operand, { ",", x86Operand } ;
x86Operand = [ x86OperandSizeOverride ], [ x86OperandType ], x86OperandKind ;
x86OperandSizeOverride = "byte" | "word" | "dword" | "qword" | "tbyte" ;
x86OperandType = "byte" | "word" | "dword" | "qword" | "xmmword" | "ymmword" | "zmmword" | "ptr" | "far" | "near" | "short" | "tbyte" | "fword" | "signed" | "unsigned" | "threadhandle" ;
x86OperandKind = immediate
| x86Register
| x86MemoryOperand
| symbolReference ;
x86ModifiableOperand = [ x86OperandSizeOverride ], [ x86OperandType ], ( x86RegisterOperand | x86MemoryOperand ) ;
x86RegisterOperand = x86Register ;
x86MemoryOperand = "[", [ x86SegmentPrefix, ":" ], x86AddressBase, [ x86AddressOffset ], "]" ;
x86AddressBase = x86RegisterOperand
| symbolReference
| ( "rel", symbolReference ) ;
x86AddressOffset = x86AddressDisplacement
| x86AddressScaleIndex ;
x86AddressDisplacement = [ "+" | "-" ], x86AddressTerm, { [ "+" | "-" ], x86AddressTerm } ;
x86AddressScaleIndex = "+", x86RegisterOperand, "", x86ScaleFactor ;
x86AddressTerm = constant
| x86RegisterOperand ;
x86ScaleFactor = "1" | "2" | "4" | "8" ;
x86Register = generalRegister
| segmentRegister
| controlRegister
| debugRegister
| mmxRegister
| xmmRegister
| ymmRegister
| zmmRegister ;
```

### ARM Instructions

```ebnf
armInstruction = [ label, ":" ], [ armInstructionPrefix ], armInstructionBody ; (* @arch: arm *)
armInstructionBody = armMnemonic, [ armOperandList ]
| armShorthandInstruction ;
armShorthandInstruction = armModifiableOperand, shorthandOperator, armOperand ;
armMnemonic = [ namespaceQualifier ], armInstructionMnemonic ;
armInstructionMnemonic = "mov"
| "add"
| "ldr"
| "str"
| "b"
| "bl"
| "cmp"
| "tst"
| "and"
| "orr"
| "eor"
| "sub"
| "rsb"
| "mul"
| "mla"
| "sdiv"
| "udiv"
| "push"
| "pop"
;
armInstructionPrefix = { armConditionCode }, [ armHintPrefix ] ;
armConditionCode = "eq" | "ne" | "cs" | "cc" | "mi" | "pl" | "vs" | "vc" | "hi" | "ls" | "ge" | "lt" | "gt" | "le" | "al" ;
armHintPrefix = "wfi" | "sev" | "yield" | "nop" ;
armOperandList = armOperand, { ",", armOperand } ;
armOperand = [ armOperandSizeOverride ], armOperandKind ;
armOperandSizeOverride = "byte" | "word" | "dword" ;
armOperandKind = immediate
| armRegister
| armMemoryOperand
| symbolReference
| stringLiteral ;
armModifiableOperand = [ armOperandSizeOverride ], armRegister ;
armRegister = "r0" | "r1" | "r2" | "r3" | "r4" | "r5" | "r6" | "r7" | "r8" | "r9" | "r10" | "r11" | "r12" | "sp" | "lr" | "pc"
| "s0" | "s1" | "s2" | "s3" | "s4" | "s5" | "s6" | "s7" | "s8" | "s9" | "s10" | "s11" | "s12" | "s13" | "s14" | "s15"
| "d0" | "d1" | "d2" | "d3" | "d4" | "d5" | "d6" | "d7" | "d8" | "d9" | "d10" | "d11" | "d12" | "d13" | "d14" | "d15"
| "q0" | "q1" | "q2" | "q3" | "q4" | "q5" | "q6" | "q7" | "q8" | "q9" | "q10" | "q11" | "q12" | "q13" | "q14" | "q15"
| "apsr" | "cpsr" | "spsr" ;
armMemoryOperand = "[", armAddress ;
armAddress = armAddressBase, [ ",", armAddressOffset ], "]"
| armAddressBase, "]" ;
armAddressBase = armRegister
| symbolReference
| ( "rel", symbolReference ) ;
armAddressOffset = armAddressDisplacement
| armAddressScaleIndex
| armShiftedRegister ;
armAddressDisplacement = [ "+" | "-" ], addressTerm, { [ "+" | "-" ], addressTerm } ;
armAddressScaleIndex = armRegister, ",", shiftOperation ;
armShiftedRegister = armRegister, ",", shiftType, expression ;
shiftType = "lsl" | "lsr" | "asr" | "ror" | "rrx" ;
shiftOperation = shiftType, " ", expression ;
```

### GPU Instructions

```ebnf
gpuInstruction = [ label, ":" ], [ gpuInstructionPrefix ], gpuInstructionBody ; (* @arch: gpu *)
gpuInstructionBody = gpuMnemonic, [ gpuOperandList ]
| gpuShorthandInstruction ;
gpuShorthandInstruction = gpuModifiableOperand, shorthandOperator, gpuOperand ;
gpuMnemonic = [ namespaceQualifier ], gpuInstructionMnemonic ;

(* Example CUDA PTX-like mnemonics *)
gpuInstructionMnemonic = "mov.b32"    (* @arch: gpu, Move 32-bit data *)
| "mov.b64"    (* @arch: gpu, Move 64-bit data *)
| "mov.f32"    (* @arch: gpu, Move 32-bit float *)
| "mov.f64"    (* @arch: gpu, Move 64-bit float *)
| "ld.global.f32"  (* @arch: gpu, Load 32-bit float from global memory *)
| "ld.global.b32"  (* @arch: gpu, Load 32-bit data from global memory *)
| "st.global.b32"  (* @arch: gpu, Store 32-bit data to global memory *)
| "st.global.f32"  (* @arch: gpu, Store 32-bit float to global memory *)
| "add.s32"   (* @arch: gpu, 32-bit signed integer addition *)
| "sub.f32"    (* @arch: gpu, 32-bit float subtraction *)
| "mul.f32"  (* @arch: gpu, 32-bit float multiplication *)
| "mad.f32"     (* @arch: gpu, Multiply-add, 32-bit float *)
| "setp.eq.s32"   (* @arch: gpu, Set predicate on 32-bit signed integer equality *)
| "bra"     (* @arch: gpu, Branch *)
| "bra.uni"  (* @arch: gpu, Uniform branch (all threads take the same path) *)
;

gpuInstructionPrefix = {  } ; // Placeholder
gpuOperandList = gpuOperand, { ",", gpuOperand } ;
gpuOperand = [ gpuOperandSizeOverride ], gpuOperandKind ;
gpuOperandSizeOverride = "b8" | "b16" | "b32" | "b64" | "f32" | "f64" ;
gpuOperandKind = immediate
| gpuRegister
| gpuMemoryOperand
| symbolReference
| stringLiteral ;
gpuModifiableOperand = [ gpuOperandSizeOverride ], gpuRegister ;
gpuRegister = "r0" | "r1" | "r2" | "r3" | "r4" | "r5" | "r6" | "r7" | "r8" | "r9" | "r10" | "r11" | "r12" | "r13" | "r14" | "r15" (* General Purpose Registers *)
| "f0" | "f1" | "f2" | "f3" | "f4" | "f5" | "f6" | "f7" | "f8" | "f9" | "f10" | "f11" | "f12" | "f13" | "f14" | "f15" (* Floating Point Registers *)
| "p0" | "p1" | "p2" | "p3" | "p4" | "p5" | "p6" | "p7" (* Predicate Registers *)
| "%tid.x" | "%tid.y" | "%tid.z" (* Thread ID *)
| "%ctaid.x" | "%ctaid.y" | "%ctaid.z" (* Block ID *)
| "%ntid.x" | "%ntid.y" | "%ntid.z" (* Number of threads in a block *)
;
gpuMemoryOperand = "[", gpuAddress, "]" ;
gpuAddress = [ gpuAddressSpace ], gpuAddressExpression ;
gpuAddressSpace = "global" | "shared" | "local" | "const" ; (* Example address spaces *)
gpuAddressExpression = gpuRegister | ( gpuRegister, "+", immediate ) | ( gpuRegister, "+", gpuRegister ) | symbolReference | expression ;
```

## Directives

```ebnf
directive = ".", directiveName, [ directiveArgumentList ] ;
directiveName = dataDirective
| equateDirective
| constDirective
| incbinDirective
| timesDirective
| segmentDirective
| useDirective
| typeDirective
| mutexDirective
| conditionDirective
| globalDirective
| externDirective
| alignDirective
| sectionDirective
| ifDirective
| elifDirective
| elseDirective
| endifDirective
| ifdefDirective
| ifndefDirective
| elifdefDirective
| elifndefDirective
| entryPointDirective
| callingConventionDirective
| acpiDirective
| ioDirective
| structDirective
| cpuDirective
| bitsDirective
| stackDirective
| warningDirective
| errorDirective
| includeDirective
| includeOnceDirective
| listDirective
| nolistDirective
| debugDirective
| orgDirective
| mapDirective
| argDirective
| localDirective
| setDirective
| unsetDirective
| assertDirective
| optDirective
| evalDirective
| repDirective
| defaultDirective
| exportDirective
| commonDirective
| fileDirective
| lineDirective
| contextDirective
| endcontextDirective
| allocDirective
| freeDirective
| bitfieldDirective
| gpuDirective
| uefiDirective
| staticDirective
| dataBlockDirective
| gdtDirective
| idtDirective
| linkerDirective ;
```
### ARM Directives

```ebnf
armDirective = ".", armDirectiveName, [ armDirectiveArgumentList ] ; (* @arch: arm *)
armDirectiveName = "syntax"
| "arch"
| "thumb"
| "arm"
| "global"
| "func"
| "endfunc"
| "type"
| "size"
;
armDirectiveArgumentList = directiveArgument, { ",", directiveArgument } ;
```

## Macros

```ebnf
macroDefinition = "#macro", identifier, [ "(", parameterList, ")" ], "{", { topLevelElement }, "}" ;
```
### ARM Macros

```ebnf
armMacroDefinition = "#arm_macro", identifier, [ "(", armParameterList, ")" ], "{", { topLevelElement }, "}" ;
armParameterList = armParameter, { ",", armParameter } ;
armParameter = identifier ;
```
## Modules

```ebnf
moduleDefinition = "%module", identifier, "{", { topLevelElement }, "}" ;
```

## Register Classes

```ebnf
registerClassDefinition = "%regclass", identifier, "=", "{", registerList, "}" ;
registerList = register, { ",", register } ;
```

## Templates

```ebnf
templateDefinition = "template", [ "<", templateParameterList, ">" ], identifier, [ "(", parameterList, ")" ], [ "->", typeReference ], [ requiresClause ], [ "{", attributeList, "}" ], "{", { templateElement }, "}" ;
templateParameterList = templateParameter, { ",", templateParameter } ;
templateParameter = ( "typename", identifier, [ "requires", conceptReference ] )
| ( "const", identifier, ":", typeReference, [ "=", constExpression ] )
| ( "...", identifier ) ;
requiresClause = "requires", conceptConjunction ;
conceptConjunction = conceptDisjunction, { "&&", conceptDisjunction } ;
conceptDisjunction = conceptReference, { "||", conceptReference } ;
conceptReference = [ namespaceQualifier ], identifier, [ "<", [templateArgumentList] ,">" ] ;
templateElement = topLevelElement
| unsafeBlock
| staticBlock ;
unsafeBlock = "unsafe", "{", { topLevelElement }, "}" ;
staticBlock = "static", "{", { dataDefinition }, "}" ;
templateCall = [ namespaceQualifier ], identifier, "<", [ templateArgumentList ], ">" ;
templateArgumentList = templateArgument, { ",", templateArgument } ;
templateArgument = typeReference
| constExpression ;
```

## Comments

```ebnf
comment = ";", { commentChar } ;
commentChar = /./ ;
lineEnd = "\n" | eof ;
```

## Labels

```ebnf
label = identifier ;
```

## Shorthand Operations

```ebnf
shorthandOperator = "=" | "+=" | "-=" | "=" | "/=" | "&=" | "|=" | "^=" | "++" | "--" ;
```

## Thread Operations

```ebnf
threadCreation = "thread", [ identifier, "=" ], templateCall, [ "(", [ expressionList ] , ")" ] ;
expressionList = expression, { ",", expression } ;
threadDirective = threadJoinDirective
| threadTerminateDirective
| threadSleepDirective
| threadYieldDirective ;
threadJoinDirective = "threadjoin", identifier ;
threadTerminateDirective = "threadterminate" ;
threadSleepDirective = "threadsleep", constExpression ;
threadYieldDirective = "threadyield" ;
threadLocalDirective = "threadlocal", identifier, ":", typeReference, [ "=", constExpression ] ;
```

## Operands

```ebnf
operand = [ operandSizeOverride ], [ operandType ], operandKind ;

(* ARM Operands are defined above *)

(* GPU Operands are defined above *)

(* X86 Operands are defined above *)

modifiableOperand = [ operandSizeOverride ], [ operandType ], ( registerOperand | memoryOperand ) ;
armModifiableOperand = [ armOperandSizeOverride ], armRegister ;
gpuModifiableOperand = [ gpuOperandSizeOverride ], gpuRegister ;
x86ModifiableOperand = [ x86OperandSizeOverride ], [ x86OperandType ], ( x86RegisterOperand | x86MemoryOperand ) ;
```

### Operand Kinds

```ebnf
immediate = constant ;
registerOperand = register ;
memoryOperand = memoryAddress ;
```

### Registers

```ebnf
register = generalRegister
| segmentRegister
| controlRegister
| debugRegister
| mmxRegister
| xmmRegister
| ymmRegister
| zmmRegister
| armRegister  // Include ARM registers
| gpuRegister ; // Include GPU registers

generalRegister = "al" | "ah" | "ax" | "eax" | "rax" | "bl" | "bh" | "bx" | "ebx" | "rbx" | "cl" | "ch" | "cx" | "ecx" | "rcx" | "dl" | "dh" | "dx" | "edx" | "rdx" | "si" | "esi" | "rsi" | "di" | "edi" | "rdi" | "sp" | "esp" | "rsp" | "bp" | "ebp" | "rbp" | "r8b" | "r8w" | "r8d" | "r8" | "r9b" | "r9w" | "r9d" | "r9" | "r10b" | "r10w" | "r10d" | "r10" | "r11b" | "r11w" | "r11d" | "r11" | "r12b" | "r12w" | "r12d" | "r12" | "r13b" | "r13w" | "r13d" | "r13" | "r14b" | "r14w" | "r14d" | "r14" | "r15b" | "r15w" | "r15d" | "r15" ;
segmentRegister = "cs" | "ds" | "es" | "fs" | "gs" | "ss" ;
controlRegister = "cr0" | "cr2" | "cr3" | "cr4" | "cr8" ;
debugRegister = "dr0" | "dr1" | "dr2" | "dr3" | "dr4" | "dr5" | "dr6" | "dr7" ;
mmxRegister = "mm", digit ;
xmmRegister = "xmm", ( digit | ( "1", digit) | ( "2", digit) | ( "3", ( "0" | "1" )) ) ;
ymmRegister = "ymm", ( digit | ( "1", digit) | ( "2", digit) | ( "3", ( "0" | "1" )) ) ;
zmmRegister = "zmm", ( digit | ( "1", digit) | ( "2", digit) | ( "3", ( "0" | "1" )) ) ;
```

## Constants

```ebnf
constant = [ "-" ], ( number | hexNumber | binNumber | floatNumber | character | addressLiteral ) ;
number = digit, { digit } ;
hexNumber = ( "0x" | "0X" ), hexDigit, { hexDigit } ;
binNumber = ( "0b" | "0B" ), binDigit, { binDigit } ;
floatNumber = digit, { digit }, ".", { digit }, [ ( "e" | "E" ), [ "+" | "-" ], digit, { digit } ] ;
character = "'", ( escapeSequence | characterChar ), "'" ;
escapeSequence = "", ( "n" | "r" | "t" | """ | "'" | "`" | "x", hexDigit, hexDigit ) ;
characterChar = /[^'\\\n]/ ;
addressLiteral = "$", hexNumber ;
```

## Expressions

```ebnf
expression = conditionalExpression ;
conditionalExpression = logicalOrExpression, [ "?", expression, ":", expression ] ;
logicalOrExpression = logicalAndExpression, { "||", logicalAndExpression } ;
logicalAndExpression = bitwiseOrExpression, { "&&", bitwiseOrExpression } ;
bitwiseOrExpression = bitwiseXorExpression, { "|", bitwiseXorExpression } ;
bitwiseXorExpression = bitwiseAndExpression, { "^", bitwiseAndExpression } ;
bitwiseAndExpression = shiftExpression, { "&", shiftExpression } ;
shiftExpression = additiveExpression, { ( "<<", ">>" ), additiveExpression } ;
additiveExpression = multiplicativeExpression, { ( "+", "-" ), multiplicativeExpression } ;
multiplicativeExpression = unaryExpression, { ( "", "/", "%" ), unaryExpression } ;
unaryExpression = ( "(", expression, ")" )
| symbolReference
| constant
| ( "~" | "!" ), unaryExpression
| typeConversion, unaryExpression
| sizeOfExpression
| alignOfExpression
| templateCall ;
typeConversion = "byte" | "word" | "dword" | "qword" | "tbyte" | "float" | "double" | "signed" | "unsigned" ;
sizeOfExpression = "sizeof", "(", typeReference | expression, ")" ;
alignOfExpression = "alignof", "(", typeReference, ")" ;
```

## Memory Addresses

```ebnf
memoryAddress = "[", [ segmentPrefix, ":" ], addressBase, [ addressOffset ], "]" ;
addressBase = registerOperand
| symbolReference
| ( "rel", symbolReference ) ;
addressOffset = addressDisplacement
| addressScaleIndex ;
addressDisplacement = [ "+" | "-" ], addressTerm, { [ "+" | "-" ], addressTerm } ;
addressScaleIndex = "+", registerOperand, "", scaleFactor ;
addressTerm = constant
| registerOperand ;
scaleFactor = "1" | "2" | "4" | "8" ;
```

## String Literals

```ebnf
stringLiteral = """", { stringChar | escapeSequence }, """" ;
stringChar = /[^"\\\n]/ ;
```

## Lexical Tokens

```ebnf
identifier = /[_a-zA-Z][_a-zA-Z0-9]*/ ;
digit = /[0-9]/ ;
hexDigit = /[0-9a-fA-F]/ ;
binDigit = /[01]/ ;
eof = /$/ ;
```
