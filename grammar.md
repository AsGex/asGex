
```markdown
# AsGex Assembly Language Grammar 

**Document Version:** 1.9
**Last Updated:** 2024-10-30
**Authors:** xxx 

This document defines the grammar for the **AsGex** Assembly Language, a language designed for **zero-overhead abstractions**, **memory-safe templates**, and **compile-time metaprogramming**. It targets multiple architectures including **x64, x86-32, ARM, GPU**, and **UEFI/BIOS** environments.

---

## Table of Contents

1. [Lexical Tokens](#1-lexical-tokens)
2. [Program Structure](#2-program-structure)
3. [Namespaces](#3-namespaces)
4. [Concepts](#4-concepts)
5. [Threads](#5-threads)
6. [Procedures/Functions](#6-proceduresfunctions)
7. [Instructions](#7-instructions)
    *   [7.1 X86 Instructions](#71-x86-instructions)
    *   [7.2 ARM Instructions](#72-arm-instructions)
    *   [7.3 GPU Instructions](#73-gpu-instructions)
8. [Directives](#8-directives)
    *   [8.1 Data Directives](#81-data-directives)
    *   [8.2 Segment and Section Directives](#82-segment-and-section-directives)
    *   [8.3 Type and Structure Directives](#83-type-and-structure-directives)
    *   [8.4 Synchronization Directives](#84-synchronization-directives)
    *   [8.5 Symbol Visibility Directives](#85-symbol-visibility-directives)
    *   [8.6 Alignment and Memory Directives](#86-alignment-and-memory-directives)
    *   [8.7 Conditional Assembly Directives](#87-conditional-assembly-directives)
    *   [8.8 Entry Point and Calling Convention Directives](#88-entry-point-and-calling-convention-directives)
    *   [8.9 Architecture and Operating System Specific Directives](#89-architecture-and-operating-system-specific-directives)
    *   [8.10 Warning and Error Directives](#810-warning-and-error-directives)
    *   [8.11 Inclusion Directives](#811-inclusion-directives)
    *   [8.12 Listing and Debug Directives](#812-listing-and-debug-directives)
    *   [8.13 Argument and Variable Directives](#813-argument-and-variable-directives)
    *   [8.14 Optimization and Evaluation Directives](#814-optimization-and-evaluation-directives)
    *   [8.15 File and Line Directives](#815-file-and-line-directives)
    *   [8.16 ARM Directives](#816-arm-directives)
9. [Macros](#9-macros)
    *   [9.1. ARM Macros](#91-arm-macros)
10. [Modules](#10-modules)
11. [Register Classes](#11-register-classes)
12. [Templates](#12-templates)
13. [Comments](#13-comments)
14. [Labels](#14-labels)
15. [Shorthand Operations](#15-shorthand-operations)
16. [Thread Operations](#16-thread-operations)
17. [Operands](#17-operands)
    *   [17.1 Operand Kinds](#171-operand-kinds)
    *   [17.2 Registers](#172-registers)
18. [Constants](#18-constants)
19. [Expressions](#19-expressions)
20. [Memory Addresses](#20-memory-addresses)
21. [String Literals](#21-string-literals)
22. [Supplementary Definitions](#22-supplementary-definitions)
23. [Architecture-Specific Blocks](#23-architecture-specific-blocks)
24. [Enums](#24-enums)
25. [Structs](#25-structs)

---

## 1. Lexical Tokens

```ebnf
identifier = letter, { letter | digit | "_" } ;
letter = "a".."z" | "A".."Z" ;
digit = "0".."9" ;
hexDigit = digit | "a".."f" | "A".."F" ;
binDigit = "0" | "1" ;
stringLiteral = '"', { stringCharacter }, '"' ;
stringCharacter = escapeSequence | /[^"\\]/ ;
escapeSequence = "\\", ( "n" | "r" | "t" | "\"" | "'" | "`" | "\\" | "0" | "x", hexDigit, hexDigit ) ;
```

## 2. Program Structure

```ebnf
program = { topLevelElement }, eof ;

topLevelElement = instructionLine
                | directiveLine
                | macroDefinition
                | templateDefinition
                | moduleDefinition
                | registerClassDefinition
                | threadDefinition
                | enumDefinition
                | structDefinition
                | threadLocalDirectiveLine
                | namespaceDefinition
                | conceptDefinition
                | procedureDefinition
                | globalVariableDefinition
                | architectureSpecificBlock
                | emptyLine ;

instructionLine = instruction, [ comment ], lineEnd ;
directiveLine = directive, [ comment ], lineEnd ;
threadLocalDirectiveLine = threadLocalDirective, [ comment ], lineEnd ;
emptyLine = lineEnd ;
lineEnd = "\n" | eof ;
```

## 3. Namespaces

```ebnf
namespaceDefinition = "namespace", identifier, "{", { topLevelElement }, "}", [ comment ], lineEnd ;
```

## 4. Concepts

```ebnf
conceptDefinition = "concept", identifier, [ "<", templateParameterList, ">" ], [ whereClause ], "{", { conceptRequirement }, "}", [ comment ], lineEnd ;
conceptRequirement = typeRequirement
                    | expressionRequirement
                    | memberRequirement;
typeRequirement = "typename", identifier, [ ":", templateParameter ] ;
expressionRequirement = "requires", expression, ";" ;
memberRequirement = "requires", identifier, ".", identifier, [ "(", [ parameterList ], ")" ], [ "->", typeReference ], ";" ;
whereClause = "where", expression ;
```

## 5. Threads

```ebnf
threadDefinition = "thread", identifier, [ "<", templateArgumentList, ">" ], [ "(", parameterList, ")" ], [ ":", typeReference ], "{", { topLevelElement }, "}", [ comment ], lineEnd ;
parameterList = parameter, { ",", parameter } ;
parameter = identifier, ":", typeReference, [ "=", expression ] ;
```

## 6. Procedures/Functions

```ebnf
procedureDefinition = "proc", identifier, [ "(", parameterList, ")" ], [ "->", typeReference ], [ callingConvention ], "{", { topLevelElement }, "}", [ comment ], lineEnd ;
callingConvention = "callingconvention", ( "cdecl" | "stdcall" | "fastcall" | identifier ) ;
```

## 7. Instructions

```ebnf
instruction = [ label, ":" ], instructionBody ;

instructionBody = architectureSpecificInstructionBody ;

architectureSpecificInstructionBody = x86InstructionBody    (* @arch: x86, x64, x86-32 *)
                                    | armInstructionBody    (* @arch: arm, armv7-a, armv8-a, arm64 *)
                                    | gpuInstructionBody    (* @arch: gpu, cuda, rocm, ... *)
                                    | emptyInstruction ;

emptyInstruction = "nop" ;

instructionPrefix = x86InstructionPrefix
                  | armInstructionPrefix
                  | gpuInstructionPrefix ;

mnemonic = [ namespaceQualifier ], identifier ;
instructionMnemonic = x86InstructionMnemonic
                      | armInstructionMnemonic
                      | gpuInstructionMnemonic ;
x86InstructionMnemonic = identifier;
armInstructionMnemonic = identifier;
gpuInstructionMnemonic = identifier;
```

### 7.1 X86 Instructions

```ebnf
x86InstructionBody = [ x86InstructionPrefix ], x86InstructionMnemonic, [ x86OperandList ], [ comment ], lineEnd   (* @arch: x86, x64, x86-32 *)
                   | x86ShorthandInstruction, [ comment ], lineEnd ;
x86ShorthandInstruction = x86ModifiableOperand, shorthandOperator, expression ;
x86InstructionPrefix = { x86Prefix } ;
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
                | x86RegisterOperand
                | x86MemoryOperand
                | symbolReference ;
x86ModifiableOperand = [ x86OperandSizeOverride ], [ x86OperandType ], ( x86RegisterOperand | x86MemoryOperand ) ;
x86RegisterOperand = x86Register ;
x86MemoryOperand = "[", [ x86SegmentPrefix, ":" ], x86AddressExpression, "]" ;
x86AddressExpression = x86AddressBase, [ x86AddressOffset ] ;
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

### 7.2 ARM Instructions

```ebnf
armInstruction = [ label, ":" ], armInstructionBody ;
armInstructionBody = [ armInstructionPrefix ], armInstructionMnemonic, [ armOperandList ], [ comment ], lineEnd (* @arch: arm, armv7-a, armv8-a, arm64 *)
                   | armShorthandInstruction, [ comment ], lineEnd ;
armShorthandInstruction = armModifiableOperand, shorthandOperator, armOperand ;
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

### 7.3 GPU Instructions

```ebnf
gpuInstructionBody = [ gpuInstructionPrefix ], gpuInstructionMnemonic, [ gpuOperandList ], [ comment ], lineEnd (* @arch: gpu *)
                   | gpuShorthandInstruction, [ comment ], lineEnd ;
gpuShorthandInstruction = gpuModifiableOperand, shorthandOperator, gpuOperand ;
gpuInstructionPrefix = { "activemask" | "predicated" };
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
gpuAddressSpace = "global" | "shared" | "local" | "const" ;
gpuAddressExpression = gpuRegister
                     | ( gpuRegister, "+", immediate )
                     | ( gpuRegister, "+", gpuRegister )
                     | symbolReference
                     | expression ;
```

## 8. Directives

```ebnf
directive = dataDefinition
          | equateDefinition
          | constDefinition
          | incbinDefinition
          | timesDefinition
          | segmentDefinition
          | useDefinition
          | typeDefinition
          | mutexDefinition
          | conditionDefinition
          | globalDefinition
          | externDefinition
          | alignDefinition
          | sectionDefinition
          | ifDirective
          | elifDirective
          | elseDirective
          | endifDirective
          | ifdefDirective
          | ifndefDirective
          | elifdefDirective
          | elifndefDirective
          | entryPointDefinition
          | callingConventionDefinition
          | acpiDefinition
          | ioDefinition
          | structDefinition
          | cpuDefinition
          | bitsDefinition
          | stackDefinition
          | warningDefinition
          | errorDefinition
          | includeDefinition
          | includeOnceDefinition
          | listDefinition
          | nolistDefinition
          | debugDefinition
          | orgDefinition
          | mapDefinition
          | argDefinition
          | localDefinition
          | setDefinition
          | unsetDefinition
          | assertDefinition
          | optDefinition
          | evalDefinition
          | repDirective
          | defaultDirective
          | exportDefinition
          | commonDefinition
          | fileDefinition
          | lineDefinition
          | contextDefinition
          | endcontextDefinition
          | allocDefinition
          | freeDefinition
          | bitfieldDefinition
          | gpuDefinition
          | uefiDefinition
          | staticDefinition
          | dataBlockDefinition
          | gdtDefinition
          | idtDefinition
          | linkerDefinition
          | armDirective
          ;

dataDirective = "db" | "dw" | "dd" | "dq" | "dt" | "do" | "ddq" | "dqq" | "dhq" | "dvq" | "resb" | "resw" | "resd" | "resq" | "rest" | "reso" | "resdq" | "resqq" | "reshq" | "resvq";
equateDirective = "equ" ;
constDirective = "const" ;
incbinDirective = "incbin" ;
timesDirective = "times" ;
segmentDirective = "segment" | "section" ;
useDirective = "use" ;
typeDirective = "type" ;
mutexDirective = "mutex" ;
conditionDirective = "condition" ;
globalDirective = "global" ;
externDirective = "extern" ;
alignDirective = "align" ;
sectionDirective = "section";
ifDirective = "if" ;
elifDirective = "elif" ;
elseDirective = "else" ;
endifDirective = "endif" ;
ifdefDirective = "ifdef" ;
ifndefDirective = "ifndef" ;
elifdefDirective = "elifdef" ;
elifndefDirective = "elifndef" ;
entryPointDirective = "entry" ;
callingConventionDirective = "callingconvention" ;
acpiDirective = "acpi" ;
ioDirective = "io" ;
structDirective = "struct" ;
cpuDirective = "cpu" ;
bitsDirective = "bits" ;
stackDirective = "stack" ;
warningDirective = "warning" ;
errorDirective = "error" ;
includeDirective = "include" ;
includeOnceDirective = "includeonce" ;
listDirective = "list" ;
nolistDirective = "nolist" ;
debugDirective = "debug" ;
orgDirective = "org" ;
mapDirective = "map" ;
argDirective = "arg" ;
localDirective = "local";
setDirective = "set" ;
unsetDirective = "unset" ;
assertDirective = "assert" ;
optDirective = "opt" ;
evalDirective = "eval" ;
repDirective = "rep";
defaultDirective = "default";
exportDirective = "export";
commonDirective = "common";
fileDirective = "file";
lineDirective = "line";
contextDirective = "context";
endcontextDirective = "endcontext";
allocDirective = "alloc";
freeDirective = "free";
bitfieldDirective = "bitfield";
gpuDirective = "gpu" ;
uefiDirective = "uefi";
staticDirective = "static";
dataBlockDirective = "datablock";
gdtDirective = "gdt";
idtDirective = "idt";
linkerDirective = "linker";

dataDefinition = [ label, ":" ], dataDirective, { "," , constant | stringLiteral } , lineEnd ;
equateDefinition = [ label, ":" ], equateDirective, identifier, ",", expression, lineEnd;
constDefinition = [ label, ":" ], constDirective, identifier, "=", constExpression, lineEnd;
incbinDefinition = ".", incbinDirective, stringLiteral, lineEnd;
timesDefinition = ".", timesDirective, expression, instructionLine | dataDefinition, lineEnd;
segmentDefinition = ".", segmentDirective, identifier, lineEnd;
useDefinition = ".", useDirective, namespaceQualifier, lineEnd;
typeDefinition = ".", typeDirective, identifier, "=", typeReference, lineEnd;
mutexDefinition = ".", mutexDirective, identifier, lineEnd;
conditionDefinition = ".", conditionDirective, identifier, lineEnd;
globalDefinition = ".", globalDirective, identifier, lineEnd;
externDefinition = ".", externDirective, identifier, [":", typeReference], lineEnd;
alignDefinition = ".", alignDirective, expression, lineEnd;
sectionDefinition = ".", sectionDirective, identifier, lineEnd;
ifDirective = "if", conditionalExpressionLine ;
elifDirective = "elif", conditionalExpressionLine ;
elseDirective = "else", lineEnd ;
endifDirective = "endif", lineEnd ;
ifdefDirective = "ifdef", identifier, lineEnd ;
ifndefDirective = "ifndef", identifier, lineEnd ;
elifdefDirective = "elifdef", identifier, lineEnd ;
elifndefDirective = "elifndef", identifier, lineEnd ;
entryPointDefinition = ".", entryPointDirective, [identifier], lineEnd;
callingConventionDefinition = ".", callingConventionDirective, ( "cdecl" | "stdcall" | "fastcall" | identifier ), lineEnd;
acpiDefinition = ".", acpiDirective, /* specific arguments */ lineEnd;
ioDefinition = ".", ioDirective, /* specific arguments */ lineEnd;
structDefinition = "struct", identifier, [":", typeReference ], "{", { structMember }, "}", [ comment ], lineEnd ;
cpuDefinition = ".", cpuDirective, identifier, lineEnd;
bitsDefinition = ".", bitsDirective, ( "16" | "32" | "64" ), lineEnd;
stackDefinition = ".", stackDirective, expression, lineEnd;
warningDefinition = ".", warningDirective, stringLiteral, lineEnd;
errorDefinition = ".", errorDirective, stringLiteral, lineEnd;
includeDefinition = ".", includeDirective, stringLiteral, lineEnd;
includeOnceDefinition = ".", includeOnceDirective, stringLiteral, lineEnd;
listDefinition = ".", listDirective, lineEnd;
nolistDefinition = ".", nolistDirective, lineEnd;
debugDefinition = ".", debugDirective, [expression], lineEnd;
orgDefinition = ".", orgDirective, expression, lineEnd;
mapDefinition = ".", mapDirective, expression, ",", expression, lineEnd;
argDefinition = ".", argDirective, identifier, ":", typeReference, lineEnd;
localDefinition = ".", localDirective, identifier, ":", typeReference, [ "=", expression ], lineEnd;
setDefinition = ".", setDirective, identifier, "=", expression, lineEnd;
unsetDefinition = ".", unsetDirective, identifier, lineEnd;
assertDefinition = ".", assertDirective, expression, [",", stringLiteral], lineEnd;
optDefinition = ".", optDirective, ( "enable" | "disable" ), ",", identifier, lineEnd;
evalDefinition = ".", evalDirective, expression, lineEnd;
repDirective = ".", repDirective, expression, instructionLine | dataDefinition, lineEnd;
defaultDirective = ".", defaultDirective, identifier, "=", expression, lineEnd;
exportDefinition = ".", exportDirective, identifier, lineEnd;
commonDefinition = ".", commonDirective, identifier, [":", typeReference], [",", expression], lineEnd;
fileDefinition = ".", fileDirective, stringLiteral, lineEnd;
lineDefinition = ".", lineDirective, number, [",", stringLiteral], lineEnd;
contextDefinition = ".", contextDirective, stringLiteral, lineEnd;
endcontextDefinition = ".", endcontextDirective, lineEnd;
allocDefinition = ".", allocDirective, expression, lineEnd;
freeDefinition = ".", freeDirective, expression, lineEnd;
bitfieldDefinition = ".", bitfieldDirective, identifier, ":", number, lineEnd;
gpuDefinition = ".", gpuDirective, /* specific arguments */ lineEnd;
uefiDefinition = ".", uefiDirective, /* specific arguments */ lineEnd;
staticDefinition = ".", staticDirective, dataDefinition, lineEnd;
dataBlockDefinition = ".", dataBlockDirective, identifier, /* specific arguments */ lineEnd;
gdtDefinition = ".", gdtDirective, /* specific arguments */ lineEnd;
idtDefinition = ".", idtDirective, /* specific arguments */ lineEnd;
linkerDefinition = ".", linkerDirective, stringLiteral, lineEnd;

conditionalExpressionLine = expression, lineEnd;

conditionalExpression = logicalOrExpression
                      | cpuDetectionExpression
                      | armDetectionExpression;

cpuDetectionExpression = cpuBrandCheck | cpuIdCheck;
cpuBrandCheck = "cpu_brand", ("==", "!=", ), stringLiteral;

cpuIdCheck = "cpuid.", cpuIdLeaf, [ ".", cpuIdSubleaf ], ".", cpuIdRegister, ( "==", "!=", ">", "<", ">=", "<=" ), constant ;
cpuIdLeaf = hexNumber;
cpuIdSubleaf = hexNumber;
cpuIdRegister = "eax" | "ebx" | "ecx" | "edx";

armDetectionExpression = "arm_version", ("==", "!=", ">", "<", ">=", "<="), number;

cpuDetectionDirectiveName = "cpuid" | "cpu_brand";
armDetectionDirectiveName = "arm_version";

cpuidDetectionErrorDirective = ".cpuid_detection_error", ( "error" | "warning" | "ignore" ) ;
armVersionDetectionErrorDirective = ".arm_version_detection_error", ( "error" | "warning" | "ignore" ) ;

stringDirective = "db", { "," , constant | stringLiteral } ;
asciiDirective = ".ascii", { "," , stringLiteral } , lineEnd;
ascizDirective = ".asciz", { "," , stringLiteral } , lineEnd;
```

### 8.16 ARM Directives

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
armDirectiveArgumentList = armDirectiveArgument, { ",", armDirectiveArgument };
armDirectiveArgument = expression | stringLiteral | identifier;
```

## 9. Macros

```ebnf
macroDefinition = "#macro", identifier, [ "(", parameterList, ")" ], "{", { macroBodyElement }, "}", [ comment ], lineEnd ;
macroBodyElement = topLevelElement
                 | instructionLine
                 | directiveLine
                 | macroExpansion ;
macroExpansion = identifier [ "(", [ expressionList ], ")" ] ;
```

### 9.1 ARM Macros

```ebnf
armMacroDefinition = "#arm_macro", identifier, [ "(", armParameterList, ")" ], "{", { macroBodyElement }, "}", [ comment ], lineEnd ;
armParameterList = armParameter, { ",", armParameter } ;
armParameter = identifier ;
```

## 10. Modules

```ebnf
moduleDefinition = "%module", identifier, "{", { topLevelElement }, "}", [ comment ], lineEnd ;
```

## 11. Register Classes

```ebnf
registerClassDefinition = "%regclass", identifier, "=", "{", registerList, "}", [ comment ], lineEnd ;
registerList = register, { ",", register } ;
```

## 12. Templates

```ebnf
templateDefinition = "template", [ "<", templateParameterList, ">" ], identifier, [ "(", parameterList, ")" ], [ "->", typeReference ], [ requiresClause ], [ "{", attributeList, "}" ], "{", { templateElement }, "}", [ comment ], lineEnd ;
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
attributeList = identifier, { ",", identifier } ;
```

## 13. Comments

```ebnf
comment = ";", { commentChar } ;
commentChar = /./ ;
```

## 14. Labels

```ebnf
label = identifier ;
```

## 15. Shorthand Operations

```ebnf
shorthandOperator = "=" | "+=" | "-=" | "*=" | "/=" | "&=" | "|=" | "^=" | "++" | "--" ;
```

## 16. Thread Operations

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

## 17. Operands

```ebnf
operand = [ operandSizeOverride ], [ operandType ], operandKind ;
operandSizeOverride = "byte" | "word" | "dword" | "qword" | "tbyte" | "oword" | "dqword" | "qqword" | "hqqword" | "vqqword"
                    | "byte ptr" | "word ptr" | "dword ptr" | "qword ptr" ;
operandType = "byte" | "word" | "dword" | "qword" | "xmmword" | "ymmword" | "zmmword" | "b128" | "b256" | "b512" | "b1024" | "b2048" | "ptr" | "far" | "near" | "short" | "tbyte" | "fword" | "signed" | "unsigned" | "threadhandle" ;

x86Operand = [ x86OperandSizeOverride ], [ x86OperandType ], x86OperandKind ;
armOperand = [ armOperandSizeOverride ], armOperandKind ;
gpuOperand = [ gpuOperandSizeOverride ], gpuOperandKind ;

modifiableOperand = [ operandSizeOverride ], [ operandType ], ( registerOperand | memoryOperand ) ;
```

### 17.1 Operand Kinds

```ebnf
operandKind = immediate
            | registerOperand
            | memoryOperand
            | symbolReference ;
symbolReference = [ namespaceQualifier ], identifier ;
```

### 17.2 Registers

```ebnf
register = generalRegister```ebnf
         | segmentRegister
         | controlRegister
         | mmxRegister
         | xmmRegister
         | ymmRegister
         | zmmRegister
         | kRegister
         | bndRegister
         | armRegister
         | gpuRegister
         | debugRegister ;
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
xmmRegister = "xmm0" | "xmm1" | "xmm2" | "xmm3" | "xmm4" | "xmm5" | "xmm6"
            | "xmm7"
            | "xmm8" | "xmm9" | "xmm10" | "xmm11" | "xmm12" | "xmm13"
            | "xmm14" | "xmm15"
            | "xmm16" | "xmm17" | "xmm18" | "xmm19" | "xmm20" | "xmm21" | "xmm22" | "xmm23"
            | "xmm24" | "xmm25" | "xmm26" | "xmm27" | "xmm28" | "xmm29" | "xmm30" | "xmm31";
ymmRegister = "ymm0" | "ymm1" | "ymm2" | "ymm3" | "ymm4" | "ymm5" | "ymm6" | "ymm7"
            | "ymm8" | "ymm9" | "ymm10" | "ymm11" | "ymm12" | "ymm13" | "ymm14" | "ymm15"
            | "ymm16" | "ymm17" | "ymm18" | "ymm19" | "ymm20" | "ymm21" | "ymm22" | "ymm23"
            | "ymm24" | "ymm25" | "ymm26" | "ymm27" | "ymm28"
            | "ymm29" | "ymm30" | "ymm31";
zmmRegister = "zmm0" | "zmm1" | "zmm2" | "zmm3" | "zmm4" | "zmm5" | "zmm6" | "zmm7"
            | "zmm8" | "zmm9" | "zmm10" | "zmm11"
            | "zmm12" | "zmm13" | "zmm14" | "zmm15"
            | "zmm16" | "zmm17" | "zmm18" | "zmm19" | "zmm20" | "zmm21" | "zmm22" | "zmm23"
            | "zmm24" | "zmm25" | "zmm26" | "zmm27" | "zmm28" | "zmm29" | "zmm30" | "zmm31";
kRegister = "k0" | "k1" | "k2" | "k3" | "k4" | "k5" | "k6" | "k7";
bndRegister = "bnd0" | "bnd1" | "bnd2" | "bnd3";
```

## 18. Constants

```ebnf
constant = [ "-" ], constantValue ;
constantValue = number
              | hexNumber
              | binNumber
              | floatNumber
              | character
              | addressLiteral
              | cpuIdLiteral
              | armVersionLiteral;
cpuIdLiteral = "cpuid";
armVersionLiteral = "arm_version";
number = digit, { digit } ;
hexNumber = ( "0x" | "0X" ), hexDigit, { hexDigit } ;
binNumber = ( "0b" | "0B" ), binDigit, { binDigit } ;
floatNumber = digit, { digit }, ".", { digit }, [ ( "e" | "E" ), [ "+" | "-" ], digit, { digit } ] ;
character = "'", characterContent, "'" ;
characterContent = escapeSequence | characterChar ;
characterChar = /[^'\\\n]/ ;
addressLiteral = "$", hexNumber ;

constExpression = [ "-" ], constValue, { ( "+" | "-" | "*" | "/" | "%" | "<<" | ">>" | "&" | "|" | "^" ), [ "-" ], constValue } ;
constValue = number
           | hexNumber
           | binNumber
           | character
           | constSymbol ;
constSymbol = identifier ;
```

## 19. Expressions

```ebnf
expression = conditionalExpression ;
conditionalExpression = logicalOrExpression ;

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
                | cpuIdOperand
                | armVersionOperand;

cpuIdOperand = cpuIdLiteral;
armVersionOperand = armVersionLiteral;
typeConversion = "(", typeReference, ")" ;
sizeOfExpression = "sizeof", "(", typeReference | expression, ")" ;
alignOfExpression = "alignof", "(", typeReference, ")" ;
```

## 20. Memory Addresses

```ebnf
memoryOperand = architectureSpecificMemoryOperand ;

architectureSpecificMemoryOperand = x86MemoryOperand   (* @arch: x86, x64, x86-32 *)
                                 | armMemoryOperand   (* @arch: arm, armv7-a, armv8-a, arm64 *)
                                 | gpuMemoryOperand   (* @arch: gpu, cuda, rocm, ... *) ;
```

## 21. String Literals

```ebnf
stringLiteral = '"', { stringCharacter }, '"' ;
stringCharacter = escapeSequence | /[^"\\]/ ;
```

## 22. Supplementary Definitions

```ebnf
namespaceQualifier = identifier, { "::", identifier } ;
```

## 23. Architecture-Specific Blocks

```ebnf
architectureSpecificBlock = architectureSpecificTag, "{", { architectureSpecificElement }, "}", [ comment ], lineEnd ;
architectureSpecificTag = "@x86" | "@x64" | "@x86-32" | "@arm" | "@arm64" | "@armv7-a" | "@armv8-a" | "@gpu" | "@cuda" | "@rocm" | identifier ;
architectureSpecificElement = instructionLine
                            | directiveLine
                            | macroDefinition
                            | templateDefinition
                            | moduleDefinition
                            | registerClassDefinition
                            | threadDefinition
                            | enumDefinition
                            | structDefinition
                            | threadLocalDirectiveLine
                            | namespaceDefinition
                            | conceptDefinition
                            | procedureDefinition
                            | globalVariableDefinition
                            | architectureSpecificBlock ;
```

## 24. Enums

```ebnf
enumDefinition = "enum", identifier, [":", typeReference ], "{", enumItemList, "}", [ comment ], lineEnd ;
enumItemList = enumItem, { ",", enumItem }, [ "," ] ;
enumItem = identifier, [ "=", constExpression ] ;
```

## 25. Structs

```ebnf
structDefinition = "struct", identifier, [":", typeReference ], "{", { structMember }, "}", [ comment ], lineEnd ;
structMember = typeReference, identifier, [ "=", expression ], ";", [ comment ], lineEnd ;
typeReference = [ namespaceQualifier ], identifier ;
```
