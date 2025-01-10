```markdown
## Enhanced EBNF Grammar for Advanced Assembly Language (V32) - X64, Bit, GPU, UEFI/BIOS

**Focus:** Zero-Overhead, Memory-Safe Templates, and Compile-Time Metaprogramming

### Program Structure

program = v32Program [ conditionalARM ], eof ;
v32Program = { topLevelElement } ;
conditionalARM = armProgram | /* empty */ ;
armProgram = { armTopLevelElement } ;

topLevelElement = declaration
                | executableElement ;

armTopLevelElement = armInstruction
                      | armDirective
                      | armMacroDefinition ;

declaration = namespaceDefinition
            | conceptDefinition
            | threadDefinition
            | procedureDefinition
            | macroDefinition
            | templateDefinition
            | moduleDefinition
            | registerClassDefinition
            | enumDefinition
            | structDefinition ;

executableElement = instruction
                  | directive
                  | threadLocalDirective ;

### Namespaces

namespaceDefinition = "namespace", identifier, "{", { topLevelElement }, "}" ;

### Concepts

conceptDefinition = "%concept", identifier, [ "<", templateParameterList, ">" ], "{", { conceptRequirement }, "}" ;
conceptRequirement = typeRequirement
                     | expressionRequirement ;
typeRequirement = "typename", identifier, ":", templateParameter ;
expressionRequirement = "requires", expression, ";" ;

### Threads

threadDefinition = "thread", identifier, [ "<", templateArgumentList, ">" ], [ "(", parameterList, ")" ], [ ":", typeReference ], "{", { topLevelElement }, "}" ;
parameterList = parameter, { ",", parameter } ;
parameter = identifier, ":", typeReference ;

### Procedures/Functions

procedureDefinition = "proc", identifier, [ "(", parameterList, ")" ], [ "->", typeReference ], [ "callingconvention", identifier ], "{", { topLevelElement }, "}" ;

### Instructions

instruction = [ label, ":" ], [ instructionPrefix ], instructionBody, [ comment ], lineEnd ;
instructionBody = mnemonic, [ operandList ]
                   | shorthandInstruction ;
shorthandInstruction = modifiableOperand, shorthandOperator, operand ;
mnemonic = [ namespaceQualifier ], instructionMnemonic
           | templateCall ;
namespaceQualifier = identifier, "::" ;
instructionMnemonic = "mov"
                        | "add"
                        | "sub"
                        | "jmp"
                        | "call"
                        | "ret"
                        | "push"
                        | "pop"
                        | "lea"
                        | "cmp"
                        | "test"
                        | "and"
                        | "or"
                        | "xor"
                        | "not"
                        | "neg"
                        | "mul"
                        | "imul"
                        | "div"
                        | "idiv"
                        | "shl"
                        | "shr"
                        | "sar"
                        | "rol"
                        | "ror"
                        | "rcl"
                        | "rcr"
                        | "jz"
                        | "jnz"
                        | "je"
                        | "jne"
                        | "js"
                        | "jns"
                        | "jg"
                        | "jge"
                        | "jl"
                        | "jle"
                        | "ja"
                        | "jae"
                        | "jb"
                        | "jbe" ; (* Example Mnemonics *)

### ARM Instructions (Optional)

armInstruction = [ label, ":" ], [ armInstructionPrefix ], armInstructionBody, [ comment ], lineEnd ;
armInstructionBody = armMnemonic, [ armOperandList ]
                     | armShorthandInstruction ;
armShorthandInstruction = armModifiableOperand, shorthandOperator, armOperand ;
armMnemonic = [ namespaceQualifier ], armInstructionMnemonic ;
armInstructionMnemonic = "mov" // Example ARM mnemonics
                            | "add"
                            | "ldr"
                            | "str"
                            // ... more ARM mnemonics ...
                            ;

### Directives

directive = ".", directiveName, [ directiveArguments ], [ comment ], lineEnd ;
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
                | ifdefDirective
                | ifndefDirective
                | elifdefDirective
                | elifndefDirective
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

### ARM Directives (Optional)

armDirective = ".", armDirectiveName, [ armDirectiveArguments ], [ comment ], lineEnd ;
armDirectiveName = "syntax"  // Example ARM specific directives
                   | "arch"
                   | "thumb"
                   | "arm"
                   // ... more ARM directives ...
                   ;
armDirectiveArguments = directiveArgument, { ",", directiveArgument } ;

### Linker Directives

linkerDirective = "library", stringLiteral ;

### GDT/IDT Directives

gdtDirective = "gdt", identifier, "{", { dataDefinition }, "}" ;
idtDirective = "idt", identifier, "{", { dataDefinition }, "}" ;

### GPU/UEFI Specific Directives

gpuDirective = "gpu", identifier, [ "(", [ directiveArgumentList ], ")" ] ;
uefiDirective = "uefi", identifier, [ "(", [ directiveArgumentList ], ")" ] ;

### Static Directive

staticDirective = "static", [ sectionSpecifier ], dataDefinition ;
sectionSpecifier = "[", identifier, "]" ;
dataDefinition = identifier, ":", typeReference, [ "=", constExpression ] ;

### Data Block Directive

dataBlockDirective = "data", identifier, "{", { dataBlockItem }, "}" ;
dataBlockItem = identifier, ":", typeReference, [ "=", constExpression ], ";" ;

### Common Directive Components

directiveArguments = directiveArgument, { ",", directiveArgument } ;
directiveArgument = stringLiteral
                  | integerLiteral
                  | hexNumber
                  | binNumber
                  | floatNumber
                  | [ namespaceQualifier ], identifier
                  | expression
                  | character ;
directiveArgumentList = directiveArgument, { ",", directiveArgument } ;

### Data Directives

dataDirective = ( "db" | "dw" | "dd" | "dq" | "dt" | "resb" | "resw" | "resd" | "resq" | "rest" | "string" ), dataList ;
dataList = dataValue, { ",", dataValue } ;
dataValue = stringLiteral
            | expression ;

### Type Directives

typeDirective = "type", identifier, "as", typeDefinition ;
typeDefinition = basicType
                 | arrayType
                 | structReference
                 | enumReference
                 | pointerType
                 | templateReference ;
basicType = ( "byte" | "word" | "dword" | "qword" | "tbyte" | "float" | "double" | "string" ), [ "signed" | "unsigned" ] ;
arrayType = "array", "<", typeReference, ",", arraySizeExpression, ">", [ "checked" ] ;
arraySizeExpression = expression ;
pointerType = "ptr", "<", typeReference, [ ",", ( "mutable" | "immutable" ) ], ">" ;
typeReference = [ namespaceQualifier ], identifier
              | templateCall ;
structReference = [ namespaceQualifier ], identifier ;
enumReference = [ namespaceQualifier ], identifier ;
templateReference = [ namespaceQualifier ], identifier, "<", [ typeReferenceList ], ">" ;
typeReferenceList = typeReference, { ",", typeReference } ;

### Enum Definition

enumDefinition = "%enum", identifier, [ ":", typeReference ], "{", enumMemberList, "}" ;
enumMemberList = enumMember, { ",", enumMember } ;
enumMember = identifier, [ "=", constExpression ] ;

### Struct Definition

structDefinition = "%struct", identifier, [ "{", attributeList, "}" ], "{", structMemberList, "}" ;
structMemberList = structMember, { ";", structMember } ;
structMember = identifier, ":", typeReference, [ "=", expression ], [ "{", attributeList, "}" ] ;

### Bitfield Directive & Definition

bitfieldDirective = "%bitfield", identifier, [ ":", typeReference ], "{", bitfieldMemberList, "}" ;
bitfieldMemberList = bitfieldMember, { ";", bitfieldMember } ;
bitfieldMember = identifier, ":", typeReference, ":", constExpression ;

### Attributes

attributeList = attribute, { ",", attribute } ;
attribute = identifier, [ "(", [ constExpressionList ], ")" ] ;
constExpressionList = constExpression, { ",", constExpression } ;

### Other Directives

equateDirective = ".equ", identifier, expression ;
constDirective = "const", identifier, "=", constExpression ;
constExpression = expression ;
useDirective = "use", identifier, [ "as", identifier ] ;
incbinDirective = "incbin", stringLiteral, [ ",", expression, [ ",", expression ] ] ;
timesDirective = "times", constExpression, repeatableElement ; (* More specific scope *)
repeatableElement = instruction
                    | dataDefinition
                    | dataBlockItem ;
segmentDirective = "%segment", identifier, [ ",", expression ], "{", { topLevelElement }, "}" ;
mutexDirective = "mutex", identifier ;
conditionDirective = "condition", identifier ;
globalDirective = "global", symbol, { ",", symbol } ;
externDirective = "extern", symbol, { ",", symbol } ;
symbol = [ namespaceQualifier ], identifier, ":", typeReference ;
alignDirective = "align", constExpression ;
sectionDirective = "section", identifier, [ ",", stringLiteral ] ;
ifDirective = "%if", constExpression, "{", { topLevelElement }, "}", { "%elif", constExpression, "{", { topLevelElement }, "}" }, [ "%else", "{", { topLevelElement }, "}" ], "%endif" ;
elifDirective = "%elif", constExpression, "{", { topLevelElement }, "}" ; (* Explicit definition for clarity *)
elseDirective = "%else", "{", { topLevelElement }, "}" ;       (* Explicit definition for clarity *)
endifDirective = "%endif" ;
entryPointDirective = "entrypoint", identifier ;
callingConventionDirective = "callingconvention", identifier ;
acpiDirective = "acpi", identifier, "{", { dataDefinition }, "}" ;
ioDirective = "io", ( "in" | "out" ), ( "b" | "w" | "d" ), ",", operand, ",", operand ;
cpuDirective = "cpu", identifier, { ",", identifier } ;
bitsDirective = "bits", ( "16" | "32" | "64" ) ;
stackDirective = "stack", constExpression ;
warningDirective = "warning", stringLiteral ; (* Standardized naming *)
errorDirective = "error", stringLiteral ;   (* Standardized naming *)
includeDirective = "include", stringLiteral ;
includeOnceDirective = "includeonce", stringLiteral ;
listDirective = "list" ;
nolistDirective = "nolist" ;
debugDirective = "debug", stringLiteral ;
orgDirective = "org", constExpression ;
mapDirective = "map", constExpression, ",", constExpression ;
argDirective = "arg", identifier, [ ":", typeReference ] ;
localDirective = "local", identifier, [ ":", typeReference ], [ "=", ( expression | "{", dataList, "}" ) ] ;
setDirective = "set", identifier, ( stringLiteral | expression ) ;
unsetDirective = "unset", identifier ;
assertDirective = "assert", constExpression, [ ",", stringLiteral ] ;
optDirective = "opt", identifier, { ",", identifier } ;
evalDirective = "eval", expression ; (* Shortened for consistency *)
repDirective = "rep", constExpression, repeatableElement ; (* More specific scope *)
defaultDirective = "default", identifier, "=", constExpression ;
exportDirective = [ namespaceQualifier ], identifier ;
commonDirective = "common", identifier, ",", constExpression ;
fileDirective = ".file", stringLiteral ;
lineDirective = ".line", constExpression ;
contextDirective = "context" ;
endcontextDirective = "endcontext" ;
allocDirective = "alloc", identifier, ":", typeReference, [ ",", constExpression ] ;
freeDirective = "free", modifiableOperand ;
ifdefDirective = "%ifdef", identifier ;
ifndefDirective = "%ifndef", identifier ;
elifdefDirective = "%elifdef", identifier ;
elifndefDirective = "%elifndef", identifier ;

### ARM Macros (Optional)

armMacroDefinition = "#macro", identifier, [ "(", armParameterList, ")" ], "{", { armTopLevelElement }, "}" ;
armParameterList = armParameter, { ",", armParameter } ;
armParameter = identifier, [ ":" , typeReference ] ; // Type info is optional for ARM macros

### Macros

macroDefinition = "#macro", identifier, [ "(", parameterList, ")" ], "{", { topLevelElement }, "}" ;

### Modules

moduleDefinition = "%module", identifier, "{", { topLevelElement }, "}" ;

### Register Classes

registerClassDefinition = "%regclass", identifier, "=", "{", registerList, "}" ;
registerList = register, { ",", register } ;

### Templates

templateDefinition = "template", [ "<", templateParameterList, ">" ], identifier, [ "(", parameterList, ")" ], [ "->", typeReference ], [ requiresClause ], [ "{", attributeList, "}" ], "{", { templateElement }, "}" ;
templateParameterList = templateParameter, { ",", templateParameter } ;
templateParameter = ( "typename", identifier, [ "requires", conceptReference ] )
                    | ( "const", identifier, ":", typeReference, [ "=", constExpression ] )
                    | ( "...", identifier ) ;
requiresClause = "requires", conceptConjunction ;
conceptConjunction = conceptDisjunction, { "&&", conceptDisjunction } ;
conceptDisjunction = conceptReference, { "||", conceptReference } ;
conceptReference = [ namespaceQualifier ], identifier, [ "<", [ templateArgumentList ] ,">"] ;
templateElement = topLevelElement
                  | unsafeBlock
                  | staticBlock ;
unsafeBlock = "unsafe", "{", { topLevelElement }, "}" ;
staticBlock = "static", "{", { dataDefinition }, "}" ;
templateCall = [ namespaceQualifier ], identifier, "<", [ templateArgumentList ], ">" ;
templateArgumentList = templateArgument, { ",", templateArgument } ;
templateArgument = typeReference
                   | constExpression ;

### Comments

comment = ";", { commentChar } ;
commentChar = /.+/ ;
lineEnd = "\n" | eof ;

### Labels

label = identifier ;

### Instruction Prefixes

instructionPrefix = { repeatPrefix }, { segmentPrefix }, { addressPrefix }, { dataPrefix }, { vectorPrefix }, { otherPrefix } ;
repeatPrefix = "rep" | "repe" | "repz" | "repne" | "repnz" | "lock" ;
segmentPrefix = "cs" | "ds" | "es" | "fs" | "gs" | "ss" ;
addressPrefix = "addr16" | "addr32" | "addr64" ;
dataPrefix = "byte" | "word" | "dword" | "qword" | "tbyte" ;
vectorPrefix = "xmmword" | "ymmword" | "zmmword" ;
otherPrefix = "bnd" | "notrack" | "gfx" ;

### ARM Instruction Prefixes (Optional)

armInstructionPrefix = { armConditionCode }, { armHintPrefix } ; // Example ARM prefixes
armConditionCode = "eq" | "ne" | "cs" | "cc" | "mi" | "pl" | "vs" | "vc" | "hi" | "ls" | "ge" | "lt" | "gt" | "le" | "al" ;
armHintPrefix = "wfi" | "sev" ; // Example hint prefixes

### Shorthand Operations

shorthandOperator = "=" | "+=" | "-=" | "*=" | "/=" | "&=" | "|=" | "^=" | "++" | "--" ;

### Thread Operations

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

### Operands

operandList = operand, { ",", operand } ;
operand = [ operandSizeOverride ], [ operandType ], operandKind ;
operandSizeOverride = "byte" | "word" | "dword" | "qword" | "tbyte" ;
operandType = "byte" | "word" | "dword" | "qword" | "xmmword" | "ymmword" | "zmmword" | "ptr" | "far" | "near" | "short" | "tbyte" | "fword" | "signed" | "unsigned" | "threadhandle" ;
operandKind = immediate
              | registerOperand
              | memoryOperand
              | symbolReference
              | stringLiteral
              | templateCall ;

### ARM Operands (Optional)

armOperandList = armOperand, { ",", armOperand } ;
armOperand = [ armOperandSizeOverride ], armOperandKind ;
armOperandSizeOverride = "byte" | "word" | "dword" ; // Example ARM size overrides
armOperandKind = immediate
                 | armRegister
                 | armMemoryOperand
                 | symbolReference
                 | stringLiteral ;

modifiableOperand = [ operandSizeOverride ], [ operandType ], ( registerOperand | memoryOperand ) ;
armModifiableOperand = [ armOperandSizeOverride ], ( armRegister | armMemoryOperand ) ; // Allow memory operands to be modified in ARM

### Operand Kinds

immediate = constant ;
registerOperand = register ;
memoryOperand = memoryAddress ;

armRegister = "r0" | "r1" | "r2" | "r3" | "r4" | "r5" | "r6" | "r7" | "r8" | "r9" | "r10" | "r11" | "r12" | "sp" | "lr" | "pc" // Example ARM registers
            | "s0" | "s1" | "s2" | "s3" | "s4" | "s5" | "s6" | "s7" | "s8" | "s9" | "s10" | "s11" | "s12" | "s13" | "s14" | "s15" // Example single-precision floating-point
            | "d0" | "d1" | "d2" | "d3" | "d4" | "d5" | "d6" | "d7" | "d8" | "d9" | "d10" | "d11" | "d12" | "d13" | "d14" | "d15" ; // Example double-precision floating-point

armMemoryOperand = "[", armAddressBase, [ armAddressOffset ], "]" ;
armAddressBase = armRegister
                 | symbolReference
                 | ( "rel", symbolReference ) ;
armAddressOffset = armAddressDisplacement
                   | armAddressScaleIndex ;
armAddressDisplacement = [ "+" | "-" ], addressTerm, { [ "+" | "-" ], addressTerm } ;
armAddressScaleIndex = "+", armRegister, "*", scaleFactor ; // Simplified for example

symbolReference = [ namespaceQualifier ], identifier ;

### Registers

register = generalRegister
           | segmentRegister
           | controlRegister
           | debugRegister
           | mmxRegister
           | xmmRegister
           | ymmRegister
           | zmmRegister ;
generalRegister = "al" | "ah" | "ax" | "eax" | "rax" | "bl" | "bh" | "bx" | "ebx" | "rbx" | "cl" | "ch" | "cx" | "ecx" | "rcx" | "dl" | "dh" | "dx" | "edx" | "rdx" | "si" | "esi" | "rsi" | "di" | "edi" | "rdi" | "sp" | "esp" | "rsp" | "bp" | "ebp" | "rbp" | "r8b" | "r8w" | "r8d" | "r8" | "r9b" | "r9w" | "r9d" | "r9" | "r10b" | "r10w" | "r10d" | "r10" | "r11b" | "r11w" | "r11d" | "r11" | "r12b" | "r12w" | "r12d" | "r12" | "r13b" | "r13w" | "r13d" | "r13" | "r14b" | "r14w" | "r14d" | "r14" | "r15b" | "r15w" | "r15d" | "r15" ;
segmentRegister = "cs" | "ds" | "es" | "fs" | "gs" | "ss" ;
controlRegister = "cr0" | "cr2" | "cr3" | "cr4" | "cr8" ;
debugRegister = "dr0" | "dr1" | "dr2" | "dr3" | "dr4" | "dr5" | "dr6" | "dr7" ;
mmxRegister = "mm" , digit ;
xmmRegister = "xmm", ( digit | ( "1", digit ) | ( "2", digit ) | ( "3", ( "0" | "1" )) ) ;
ymmRegister = "ymm", ( digit | ( "1", digit ) | ( "2", digit ) | ( "3", ( "0" | "1" )) ) ;
zmmRegister = "zmm", ( digit | ( "1", digit ) | ( "2", digit ) | ( "3", ( "0" | "1" )) ) ;

### Constants

constant = [ "-" ], ( number | hexNumber | binNumber | floatNumber | character | addressLiteral ) ;
number = digit, { digit } ;
hexNumber = ( "0x" | "0X" ), hexDigit, { hexDigit } ;
binNumber = ( "0b" | "0B" ), binDigit, { binDigit } ;
floatNumber = digit, { digit }, ".", { digit }, [ ( "e" | "E" ), [ "+" | "-" ], digit, { digit } ] ;
character = "'", ( escapeSequence | characterChar ), "'" ;
escapeSequence = "\\", ( "n" | "r" | "t" | "\\" | "'" | "\"" | "
```
