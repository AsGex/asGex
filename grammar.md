
```markdown
# AsGex Assembly Language Grammar

This document defines the grammar of the **AsGex** Assembly Language, covering program structure, instructions, directives, and expressions. AsGex is designed with a focus on **zero-overhead abstractions**, **memory-safe templates**, and **compile-time metaprogramming** capabilities, targeting a variety of architectures including **x64, x86-32, ARM, GPU, and UEFI/BIOS** environments.

## 1. Program Structure

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
              | procedureDefinition
              | globalVariableDefinition [ comment ] lineEnd ;
```

## 2. Namespaces

```ebnf
namespaceDefinition = "namespace", identifier, "{", { topLevelElement }, "}" ;
```

## 3. Concepts

```ebnf
conceptDefinition = "concept", identifier, [ "<", templateParameterList, ">" ], [ whereClause ], "{", { conceptRequirement }, "}" ;
conceptRequirement = typeRequirement
                   | expressionRequirement
                   | memberRequirement;
typeRequirement = "typename", identifier, ":", templateParameter ;
expressionRequirement = "requires", expression, ";" ;
memberRequirement = "requires", identifier, ".", identifier, [ "(", [ parameterList ], ")" ], ";" ;
whereClause = "where", expression ;
```

## 4. Threads

```ebnf
threadDefinition = "thread", identifier, [ "<", templateArgumentList, ">" ], [ "(", parameterList, ")" ], [ ":", typeReference ], "{", { topLevelElement }, "}" ;
parameterList = parameter, { ",", parameter } ;
parameter = identifier, ":", typeReference ;
```

## 5. Procedures/Functions

```ebnf
procedureDefinition = "proc", identifier, [ "(", parameterList, ")" ], [ "->", typeReference ], [ "callingconvention", callingConventionIdentifier ], "{", { topLevelElement }, "}" ;
callingConventionIdentifier = "cdecl" | "stdcall" | "fastcall" | identifier ;
```

## 6. Instructions

```ebnf
instruction = [ label, ":" ], [ instructionPrefix ], architectureSpecificInstructionBody ;

architectureSpecificInstructionBody = x86InstructionBody  (* @arch: x64, x86-32 *)
                                    | armInstructionBody  (* @arch: arm *)
                                    | gpuInstructionBody  (* @arch: gpu *) ;
```

### 6.1. X86 Instructions

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
x86InstructionPrefix = { x86Prefix };
x86Prefix = x86RepeatPrefix
            | x86SegmentPrefix
            | x86AddressPrefix
            | x86DataPrefix
            | x86VectorPrefix
            | x86OtherPrefix ;
x86RepeatPrefix = "rep" | "repe" | "repz" | "repne" | "repnz" | "lock" ;
x86SegmentPrefix = "cs" | "ds" | "es" | "fs" | "gs" | "ss" ;
x86AddressPrefix = "addr16" | "addr32" | "addr64" ;
x86DataPrefix = "byte" | "word" | "dword" | "qword" | "tbyte" ;
x86VectorPrefix = "xmmword" | "ymmword" | "zmmword" | "b128" | "b256" | "b512" | "b1024" | "b2048";
x86OtherPrefix = "bnd" | "notrack" | "gfx" ;
x86OperandList = x86Operand, { ",", x86Operand } ;
x86Operand = [ x86OperandSizeOverride ], [ x86OperandType ], x86OperandKind ;
x86OperandSizeOverride = "byte" | "word" | "dword" | "qword" | "tbyte" | "oword" | "dqword" | "qqword" | "hqqword" | "vqqword";
x86OperandType = "byte" | "word" | "dword" | "qword" | "xmmword" | "ymmword" | "zmmword" | "b128" | "b256" | "b512" | "b1024" | "b2048" | "ptr" | "far" | "near" | "short" | "tbyte" | "fword" | "signed" | "unsigned" | "threadhandle" ;
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
x86AddressScaleIndex = "+", x86RegisterOperand, "*", x86ScaleFactor ;
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
            | zmmRegister
            | kRegister
            | bndRegister;
```

### 6.2. ARM Instructions

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
                        | "vmov" (* @arch: armv7-a, armv8-a, NEON *)
                        | "vadd" (* @arch: armv7-a, armv8-a, NEON *)
                        | "vsub" (* @arch: armv7-a, armv8-a, NEON *)
                        | "vmul" (* @arch: armv7-a, armv8-a, NEON *)
                        | "vdiv" (* @arch: armv7-a, armv8-a, NEON *)
                        | "vld"  (* @arch: armv7-a, armv8-a, NEON *)
                        | "vst"  (* @arch: armv7-a, armv8-a, NEON *)
                        ;
armInstructionPrefix = { armConditionCode }, [ armHintPrefix ] ;
armConditionCode = "eq" | "ne" | "cs" | "cc" | "mi" | "pl" | "vs" | "vc" | "hi" | "ls" | "ge" | "lt" | "gt" | "le" | "al" ;
armHintPrefix = "wfi" | "sev" | "yield" | "nop" ;
armOperandList = armOperand, { ",", armOperand } ;
armOperand = [ armOperandSizeOverride ], armOperandKind ;
armOperandSizeOverride = "byte" | "word" | "dword" | "qword" | "oword" | "dqword" | "qqword" | "hqqword" | "vqqword";
armOperandKind = immediate
               | armRegister
               | armMemoryOperand
               | symbolReference
               | stringLiteral ;
armModifiableOperand = armRegister ;
armRegister = "r0" | "r1" | "r2" | "r3" | "r4" | "r5" | "r6" | "r7" | "r8" | "r9" | "r10" | "r11" | "r12" | "sp" | "lr" | "pc"
            | "s0" | "s1" | "s2" | "s3" | "s4" | "s5" | "s6" | "s7" | "s8" | "s9" | "s10" | "s11" | "s12" | "s13" | "s14" | "s15"
            | "d0" | "d1" | "d2" | "d3" | "d4" | "d5" | "d6" | "d7" | "d8" | "d9" | "d10" | "d11" | "d12" | "d13" | "d14" | "d15"
            | "q0" | "q1" | "q2" | "q3" | "q4" | "q5" | "q6" | "q7" | "q8" | "q9" | "q10" | "q11" | "q12" | "q13" | "q14" | "q15"
            | "v0" | "v1" | "v2" | "v3" | "v4" | "v5" | "v6" | "v7" | "v8" | "v9" | "v10" | "v11" | "v12" | "v13" | "v14" | "v15" | "v16" | "v17" | "v18" | "v19" | "v20" | "v21" | "v22" | "v23" | "v24" | "v25" | "v26" | "v27" | "v28" | "v29" | "v30" | "v31" (* @arch: armv7-a, armv8-a, NEON *)
            | "fpsid" | "fpscr" | "fpexc" (* @arch: armv7-a, armv8-a, NEON *)
            | "apsr" | "cpsr" | "spsr" ;
armMemoryOperand = "[", armAddress, "]" ;
armAddress = armAddressBase [ ",", armAddressOffset ] ;
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

### 6.3. GPU Instructions

```ebnf
gpuInstruction = [ label, ":" ], [ gpuInstructionPrefix ], gpuInstructionBody ; (* @arch: gpu *)
gpuInstructionBody = gpuMnemonic, [ gpuOperandList ]
                   | gpuShorthandInstruction ;
gpuShorthandInstruction = gpuModifiableOperand, shorthandOperator, gpuOperand ;
gpuMnemonic = [ namespaceQualifier ], gpuInstructionMnemonic ;

(* Example CUDA PTX-like mnemonics *)
gpuInstructionMnemonic = "mov.b32"    (* @arch: gpu, Move 32-bit data *)
                        | "mov.b64"    (* @arch: gpu, Move 64-bit data *)
                        | "mov.b128"   (* @arch: gpu, Move 128-bit data *)
                        | "mov.b256"   (* @arch: gpu, Move 256-bit data *)
                        | "mov.b512"   (* @arch: gpu, Move 512-bit data *)
                        | "mov.b1024"  (* @arch: gpu, Move 1024-bit data *)
                        | "mov.b2048"  (* @arch: gpu, Move 2048-bit data *)
                        | "mov.f32"    (* @arch: gpu, Move 32-bit float *)
                        | "mov.f64"    (* @arch: gpu, Move 64-bit float *)
                        | "ld.global.f32"  (* @arch: gpu, Load 32-bit float from global memory *)
                        | "ld.global.b32"  (* @arch: gpu, Load 32-bit data from global memory *)
                        | "ld.global.b128" (* @arch: gpu, Load 128-bit data from global memory *)
                        | "st.global.b32"  (* @arch: gpu, Store 32-bit data to global memory *)
                        | "st.global.f32"  (* @arch: gpu, Store 32-bit float to global memory *)
                        | "st.global.b128" (* @arch: gpu, Store 128-bit data to global memory *)
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
gpuOperandSizeOverride = "b8" | "b16" | "b32" | "b64" | "b128" | "b256" | "b512" | "b1024" | "b2048" | "f32" | "f64" ;
gpuOperandKind = immediate
               | gpuRegister
               | gpuMemoryOperand
               | symbolReference
               | stringLiteral ;
gpuModifiableOperand = gpuRegister ;
gpuRegister = "r0" | "r1" | "r2" | "r3" | "r4" | "r5" | "r6" | "r7" | "r8" | "r9" | "r10" | "r11" | "r12" | "r13" | "r14" | "r15" (* General Purpose Registers *)
            | "f0" | "f1" | "f2" | "f3" | "f4" | "f5" | "f6" | "f7" | "f8" | "f9" | "f10" | "f11" | "f12" | "f13" | "f14" | "f15" (* Floating Point Registers *)
            | "p0" | "p1" | "p2" | "p3" | "p4" | "p5" | "p6" | "p7" (* Predicate Registers *)
            | "v0" | "v1" | "v2" | "v3" | "v4" | "v5" | "v6" | "v7" | "v8" | "v9" | "v10" | "v11" | "v12" | "v13" | "v14" | "v15" (* Vector Registers *)
            | "%tid.x" | "%tid.y" | "%tid.z" (* Thread ID *)
            | "%ctaid.x" | "%ctaid.y" | "%ctaid.z" (* Block ID *)
            | "%ntid.x" | "%ntid.y" | "%ntid.z" (* Number of threads in a block *)
            ;
gpuMemoryOperand = "[", gpuAddress, "]" ;
gpuAddress = [ gpuAddressSpace ], gpuAddressExpression ;
gpuAddressSpace = "global" | "shared" | "local" | "const" ; (* Example address spaces *)
gpuAddressExpression = gpuRegister
                     | ( gpuRegister, "+", immediate )
                     | ( gpuRegister, "+", gpuRegister )
                     | symbolReference
                     | expression ;
```

## 7. Directives

```ebnf
directive = ".", directiveName, [ directiveArgumentList ] ;
directiveArgumentList = directiveArgument, { ",", directiveArgument };
directiveArgument = expression | stringLiteral | identifier;
```

### 7.1. Data Directives

```ebnf
dataDirective = "db" | "dw" | "dd" | "dq" | "dt" | "do" | "ddq" | "dqq" | "dhq" | "dvq" | "resb" | "resw" | "resd" | "resq" | "rest" | "reso" | "resdq" | "resqq" | "reshq" | "resvq";
equateDirective = "equ" ;
constDirective = "const" ;
incbinDirective = "incbin" ;
timesDirective = "times" ;
dataDefinition = [ label, ":" ], dataDirective, { "," , constant | stringLiteral } , lineEnd ;
stringDirective = "db";   (* Strings are defined as a sequence of bytes *)
```

### 7.2. Segment and Section Directives

```ebnf
segmentDirective = "segment" | "section" ;
useDirective = "use" ;
```

### 7.3. Type and Structure Directives

```ebnf
typeDirective = "type" ;
structDirective = "struct" ;
bitfieldDirective = "bitfield" ;
```

### 7.4. Synchronization Directives

```ebnf
mutexDirective = "mutex" ;
conditionDirective = "condition" ;
```

### 7.5. Symbol Visibility Directives

```ebnf
globalDirective = "global" ;
externDirective = "extern" ;
localDirective = "local";
exportDirective = "export";
commonDirective = "common";
staticDirective = "static";
```

### 7.6. Alignment and Memory Directives

```ebnf
alignDirective = "align" ;
orgDirective = "org";
mapDirective = "map";
allocDirective = "alloc";
freeDirective = "free";
dataBlockDirective = "datablock";
```

### 7.7. Conditional Assembly Directives

```ebnf
ifDirective = "if" ;
elifDirective = "elif" ;
elseDirective = "else" ;
endifDirective = "endif" ;
ifdefDirective = "ifdef" ;
ifndefDirective = "ifndef" ;
elifdefDirective = "elifdef" ;
elifndefDirective = "elifndef" ;
```

### 7.8. Entry Point and Calling Convention Directives

```ebnf
entryPointDirective = "entry" ;
callingConventionDirective = "callingconvention" ; (* Specifies calling convention, e.g., cdecl, stdcall *)
```

### 7.9. Architecture and Operating System Specific Directives

```ebnf
acpiDirective = "acpi" ;
ioDirective = "io" ;
cpuDirective = "cpu" ;
bitsDirective = "bits" ;
stackDirective = "stack" ;
gpuDirective = "gpu" ;
uefiDirective = "uefi";
gdtDirective = "gdt";
idtDirective = "idt";
linkerDirective = "linker";
```

### 7.10. Warning and Error Directives

```ebnf
warningDirective = "warning" ;
errorDirective = "error" ;
assertDirective = "assert";
```

### 7.11. Inclusion Directives

```ebnf
includeDirective = "include" ;
includeOnceDirective = "includeonce" ;
```

### 7.12. Listing and Debug Directives

```ebnf
listDirective = "list" ;
nolistDirective = "nolist" ;
debugDirective = "debug" ;
```

### 7.13. Argument and Variable Directives

```ebnf
argDirective = "arg" ;
setDirective = "set" ;
unsetDirective = "unset" ;
```

### 7.14. Optimization and Evaluation Directives

```ebnf
optDirective = "opt" ;
evalDirective = "eval" ;
repDirective = "rep";
defaultDirective = "default";
```

### 7.15. File and Line Directives

```ebnf
fileDirective = "file";
lineDirective = "line";
contextDirective = "context";
endcontextDirective = "endcontext";
```

### 7.16. ARM Directives

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
```

## 8. Macros

```ebnf
macroDefinition = "#macro", identifier, [ "(", parameterList, ")" ], "{", { macroBodyElement }, "}" ;
macroBodyElement = topLevelElement
                 | instruction
                 | directive
                 | macroExpansion ;
macroExpansion = identifier [ "(", [ expressionList ], ")" ] ;
```

### 8.1. ARM Macros

```ebnf
armMacroDefinition = "#arm_macro", identifier, [ "(", armParameterList, ")" ], "{", { macroBodyElement }, "}" ;
armParameterList = armParameter, { ",", armParameter } ;
armParameter = identifier ;
```

## 9. Modules

```ebnf
moduleDefinition = "%module", identifier, "{", { topLevelElement }, "}" ;
```

## 10. Register Classes

```ebnf
registerClassDefinition = "%regclass", identifier, "=", "{", registerList, "}" ;
registerList = register, { ",", register } ;
```

## 11. Templates

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
                 | expression ;
```

## 12. Comments

```ebnf
comment = ";", { commentChar } ;
commentChar = /./ ;
lineEnd = "\n" | eof ;
```

## 13. Labels

```ebnf
label = identifier ;
```

## 14. Shorthand Operations

```ebnf
shorthandOperator = "=" | "+=" | "-=" | "*=" | "/=" | "&=" | "|=" | "^=" | "++" | "--" ;
```

## 15. Thread Operations

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
threadLocalDirective = "threadlocal", identifier, ":", typeReference, [ "=", expression ] ;
```

## 16. Operands

```ebnf
operand = [ operandSizeOverride ], [ operandType ], operandKind ;

(* ARM Operands are defined above *)

(* GPU Operands are defined above *)

(* X86 Operands are defined above *)

modifiableOperand = [ operandSizeOverride ], [ operandType ], ( registerOperand | memoryOperand ) ;
```

### 16.1. Operand Kinds

```ebnf
operandKind = immediate
            | registerOperand
            | memoryOperand
            | symbolReference ;
```

### 16.2. Registers

```ebnf
register = generalRegister
         | segmentRegister
         | controlRegister
         | debugRegister
         | mmxRegister
         | xmmRegister
         | ymmRegister
         | zmmRegister
         | kRegister
         | bndRegister
         | armRegister
         | gpuRegister ;
```

```ebnf
generalRegister = "al" | "ah" | "ax" | "eax" | "rax" 
                | "bl" | "bh" | "bx" | "ebx" | "rbx" 
                | "cl" | "ch" | "cx" | "ecx" | "rcx" 
                | "dl" | "dh" | "dx" | "edx" | "rdx" 
                | "si" | "esi" | "rsi" 
                | "di" | "edi" | "rdi" 
                | "sp" | "esp" | "rsp" 
                | "bp" | "ebp" | "rbp" 
                | "r8b" | "r8w" | "r8d" | "r8" 
                | "r9b" | "r9w" | "r9d" | "r9" 
                | "r10b" | "r10w" | "r10d" | "r10" 
                | "r11b" | "r11w" | "r11d" | "r11" 
                | "r12b" | "r12w" | "r12d" | "r12" 
                | "r13b" | "r13w" | "r13d" | "r13" 
                | "r14b" | "r14w" | "r14d" | "r14" 
                | "r15b" | "r15w" | "r15d" | "r15" ;
segmentRegister = "cs" | "ds" | "es" | "fs" | "gs" | "ss" ;
controlRegister = "cr0" | "cr2" | "cr3" | "cr4" | "cr8" ;
debugRegister = "dr0" | "dr1" | "dr2" | "dr3" | "dr4" | "dr5" | "dr6" | "dr7" ;
mmxRegister = "mm0" | "mm1" | "mm2" | "mm3" | "mm4" | "mm5" | "mm6" | "mm7";
xmmRegister = "xmm0" | "xmm1" | "xmm2" | "xmm3" | "xmm4" | "xmm5" | "xmm6" | "xmm7" 
            | "xmm8" | "xmm9" | "xmm10" | "xmm11" | "xmm12" | "xmm13" | "xmm14" | "xmm15"
            | "xmm16" | "xmm17" | "xmm18" | "xmm19" | "xmm20" | "xmm21" | "xmm22" | "xmm23"
            | "xmm24" | "xmm25" | "xmm26" | "xmm27" | "xmm28" | "xmm29" | "xmm30" | "xmm31";
ymmRegister = "ymm0" | "ymm1" | "ymm2" | "ymm3" | "ymm4" | "ymm5" | "ymm6" | "ymm7" 
            | "ymm8" | "ymm9" | "ymm10" | "ymm11" | "ymm12" | "ymm13" | "ymm14" | "ymm15"
            | "ymm16" | "ymm17" | "ymm18" | "ymm19" | "ymm20" | "ymm21" | "ymm22" | "ymm23"
            | "ymm24" | "ymm25" | "ymm26" | "ymm27" | "ymm28" | "ymm29" | "ymm30" | "ymm31";
zmmRegister = "zmm0" | "zmm1" | "zmm2" | "zmm3" | "zmm4" | "zmm5" | "zmm6" | "zmm7" 
            | "zmm8" | "zmm9" | "zmm10" | "zmm11" | "zmm12" | "zmm13" | "zmm14" | "zmm15"
            | "zmm16" | "zmm17" | "zmm18" | "zmm19" | "zmm20" | "zmm21" | "zmm22" | "zmm23"
            | "zmm24" | "zmm25" | "zmm26" | "zmm27" | "zmm28" | "zmm29" | "zmm30" | "zmm31";
```ebnf
kRegister = "k0" | "k1" | "k2" | "k3" | "k4" | "k5" | "k6" | "k7";
bndRegister = "bnd0" | "bnd1" | "bnd2" | "bnd3";
```

## 17. Constants

```ebnf
constant = [ "-" ], ( number | hexNumber | binNumber | floatNumber | character | addressLiteral ) ;
number = digit, { digit } ;
hexNumber = ( "0x" | "0X" ), hexDigit, { hexDigit } ;
binNumber = ( "0b" | "0B" ), binDigit, { binDigit } ;
floatNumber = digit, { digit }, ".", { digit }, [ ( "e" | "E" ), [ "+" | "-" ], digit, { digit } ] ;
character = "'", ( escapeSequence | characterChar ), "'" ;
escapeSequence = "\\", ( "n" | "r" | "t" | "\"" | "'" | "`" | "\\" | "x", hexDigit, hexDigit ) ;
characterChar = /[^'\\\n]/ ;
addressLiteral = "$", hexNumber ;
```

## 18. Expressions

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
multiplicativeExpression = unaryExpression, { ( "*", "/", "%" ), unaryExpression } ;
unaryExpression = ( "(", expression, ")" )
                | symbolReference
                | constant
                | ( "~" | "!" | "+" | "-" ), unaryExpression
                | typeConversion, unaryExpression
                | sizeOfExpression
                | alignOfExpression
                | templateCall ;
typeConversion = "byte" | "word" | "dword" | "qword" | "tbyte" | "oword" | "dqword" | "qqword" | "hqqword" | "vqqword" | "float" | "double" | "signed" | "unsigned" ;
sizeOfExpression = "sizeof", "(", typeReference | expression, ")" ;
alignOfExpression = "alignof", "(", typeReference, ")" ;
templateCall = [ namespaceQualifier ], identifier, "<", [ templateArgumentList ], ">" ;
```

## 19. Memory Addresses

```ebnf
memoryAddress = "[", [ segmentPrefix, ":" ], addressExpression, "]" ;
addressExpression =  addressBase [ addressOffset ] ;
addressBase = registerOperand
            | symbolReference
            | ( "rel", symbolReference )
            | expression ;
addressOffset = addressDisplacement
              | addressScaleIndex ;
addressDisplacement = [ "+" | "-" ], addressTerm, { [ "+" | "-" ], addressTerm } ;
addressScaleIndex = "+", registerOperand, "*", scaleFactor ;
addressTerm = constant
            | registerOperand ;
scaleFactor = "1" | "2" | "4" | "8" ;
segmentPrefix = segmentRegister;
```

## 20. String Literals

```ebnf
stringLiteral = """", { stringChar | escapeSequence }, """" ;
stringChar = /[^"\\\n]/ ;
```

## 21. Lexical Tokens

```ebnf
identifier = /[_a-zA-Z][_a-zA-Z0-9]*/ ;
digit = /[0-9]/ ;
hexDigit = /[0-9a-fA-F]/ ;
binDigit = /[01]/ ;
eof = /$/ ;
whitespace = /[\t\n\r ]+/ ;
```

## Supplementary Definitions

These definitions clarify non-terminals used in the main grammar.

### `typeReference`

```ebnf
typeReference = [ namespaceQualifier, "::" ], identifier, [ "<", templateArgumentList, ">" ] ;
```

### `namespaceQualifier`

```ebnf
namespaceQualifier = identifier, { "::", identifier } ;
```

### `attributeList`

```ebnf
attributeList = attribute, { ",", attribute } ;
```

### `attribute`

```ebnf
attribute = identifier [ "(" [ expression ] ")" ] ;
```

### `register`

```ebnf
register = generalRegister
         | segmentRegister
         | controlRegister
         | debugRegister
         | mmxRegister
         | xmmRegister
         | ymmRegister
         | zmmRegister
         | armRegister
         | gpuRegister
         | kRegister
         | bndRegister;
```

### `generalRegister`

```ebnf
generalRegister = "al" | "ah" | "ax" | "eax" | "rax" 
                | "bl" | "bh" | "bx" | "ebx" | "rbx" 
                | "cl" | "ch" | "cx" | "ecx" | "rcx" 
                | "dl" | "dh" | "dx" | "edx" | "rdx" 
                | "si" | "esi" | "rsi" 
                | "di" | "edi" | "rdi" 
                | "sp" | "esp" | "rsp" 
                | "bp" | "ebp" | "rbp" 
                | "r8b" | "r8w" | "r8d" | "r8" 
                | "r9b" | "r9w" | "r9d" | "r9" 
                | "r10b" | "r10w" | "r10d" | "r10" 
                | "r11b" | "r11w" | "r11d" | "r11" 
                | "r12b" | "r12w" | "r12d" | "r12" 
                | "r13b" | "r13w" | "r13d" | "r13" 
                | "r14b" | "r14w" | "r14d" | "r14" 
                | "r15b" | "r15w" | "r15d" | "r15" ;
```

### `segmentRegister`

```ebnf
segmentRegister = "cs" | "ds" | "es" | "fs" | "gs" | "ss" ;
```

### `controlRegister`

```ebnf
controlRegister = "cr0" | "cr2" | "cr3" | "cr4" | "cr8" ;
```

### `debugRegister`

```ebnf
debugRegister = "dr0" | "dr1" | "dr2" | "dr3" | "dr4" | "dr5" | "dr6" | "dr7" ;
```

### `mmxRegister`

```ebnf
mmxRegister = "mm0" | "mm1" | "mm2" | "mm3" | "mm4" | "mm5" | "mm6" | "mm7";
```

### `xmmRegister`

```ebnf
xmmRegister = "xmm0" | "xmm1" | "xmm2" | "xmm3" | "xmm4" | "xmm5" | "xmm6" | "xmm7" 
            | "xmm8" | "xmm9" | "xmm10" | "xmm11" | "xmm12" | "xmm13" | "xmm14" | "xmm15"
            | "xmm16" | "xmm17" | "xmm18" | "xmm19" | "xmm20" | "xmm21" | "xmm22" | "xmm23"
            | "xmm24" | "xmm25" | "xmm26" | "xmm27" | "xmm28" | "xmm29" | "xmm30" | "xmm31";
```

### `ymmRegister`

```ebnf
ymmRegister = "ymm0" | "ymm1" | "ymm2" | "ymm3" | "ymm4" | "ymm5" | "ymm6" | "ymm7" 
            | "ymm8" | "ymm9" | "ymm10" | "ymm11" | "ymm12" | "ymm13" | "ymm14" | "ymm15"
            | "ymm16" | "ymm17" | "ymm18" | "ymm19" | "ymm20" | "ymm21" | "ymm22" | "ymm23"
            | "ymm24" | "ymm25" | "ymm26" | "ymm27" | "ymm28" | "ymm29" | "ymm30" | "ymm31";
```

### `zmmRegister`

```ebnf
zmmRegister = "zmm0" | "zmm1" | "zmm2" | "zmm3" | "zmm4" | "zmm5" | "zmm6" | "zmm7" 
            | "zmm8" | "zmm9" | "zmm10" | "zmm11" | "zmm12" | "zmm13" | "zmm14" | "zmm15"
            | "zmm16" | "zmm17" | "zmm18" | "zmm19" | "zmm20" | "zmm21" | "zmm22" | "zmm23"
            | "zmm24" | "zmm25" | "zmm26" | "zmm27" | "zmm28" | "zmm29" | "zmm30" | "zmm31";
```

### `kRegister`
```ebnf
kRegister = "k0" | "k1" | "k2" | "k3" | "k4" | "k5" | "k6" | "k7";
```

### `bndRegister`
```ebnf
bndRegister = "bnd0" | "bnd1" | "bnd2" | "bnd3";
```

### `templateCall`

```ebnf
templateCall = [ namespaceQualifier ], identifier, "<", [ templateArgumentList ], ">" ;
```

### `templateArgumentList`

```ebnf
templateArgumentList = templateArgument, { ",", templateArgument } ;
```

### `templateArgument`

```ebnf
templateArgument = typeReference
                 | expression ;
```

### `parameterList`

```ebnf
parameterList = parameter, { ",", parameter } ;
```

### `parameter`

```ebnf
parameter = identifier, ":", typeReference ;
```

