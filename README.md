
```markdown
# AsGex Assembly Language Documentation

This document provides a comprehensive overview of the syntax and semantics of the **AsGex** Assembly Language. **AsGex** is designed with a focus on **zero-overhead abstractions**, **memory-safe templates**, and **compile-time metaprogramming** capabilities, targeting a variety of architectures including X64, 32-bit, ARM, GPU, and UEFI/BIOS environments.

## 1. Introduction

**AsGex** aims to provide a powerful and expressive assembly-level language that transcends the limitations of traditional assembly. **Its core selling points are the ability to write low-level code with high-level abstractions that incur no runtime cost, and the power of compile-time metaprogramming for code generation and optimization.** It incorporates modern language concepts like namespaces, templates, concepts, and modules, allowing for more structured and maintainable code while retaining fine-grained control over hardware. The language emphasizes compile-time operations and incorporates features aimed at improving memory safety where possible. **AsGex's multi-architecture focus makes it suitable for a wide range of embedded, system-level, and high-performance computing tasks.**

## 2. Program Structure

An **AsGex** program consists of a sequence of top-level elements, processed sequentially by the compiler.

*   **`program`:** The top-level rule representing the entire program, composed of zero or more `topLevelElement` followed by the end-of-file marker (`eof`). Think of this as the container for all your AsGex code.
*   **`topLevelElement`:**  Represents the different kinds of elements that can appear at the top level of a program. These are the building blocks of your AsGex code. Examples include:
    *   `instructionLine`: A line containing a CPU instruction.
    *   `directiveLine`: A line containing a directive for the assembler.
    *   `macroDefinition`: Definition of a macro.
    *   `templateDefinition`: Definition of a template.
    *   `moduleDefinition`: Definition of a module.
    *   `registerClassDefinition`: Definition of a register class.
    *   `threadDefinition`: Definition of a thread.
    *   `enumDefinition`: Definition of an enumeration.
    *   `structDefinition`: Definition of a structure.
    *   `threadLocalDirectiveLine`: A line containing a thread-local directive.
    *   `namespaceDefinition`: Definition of a namespace.
    *   `conceptDefinition`: Definition of a concept.
    *   `procedureDefinition`: Definition of a procedure (function).
    *   `globalVariableDefinition`: Definition of a global variable.
    *   `architectureSpecificBlock`: A block of code specific to a certain architecture.
    *   `emptyLine`: A blank line.

*   **`instructionLine`:** An `instruction` optionally followed by a `comment` and a `lineEnd`. This is where you write your assembly instructions.
*   **`directiveLine`:** A `directive` optionally followed by a `comment` and a `lineEnd`. Directives tell the assembler how to process your code.
*   **`threadLocalDirectiveLine`:** A `threadLocalDirective` optionally followed by a `comment` and a `lineEnd`. Used for declaring variables specific to a thread.
*   **`emptyLine`:**  Consists only of a `lineEnd`. Improves readability.
*   **`lineEnd`:**  A newline character (`\n`) or the end-of-file marker (`eof`). Marks the end of a line of code.

**Example:**

```asgex
; A simple AsGex program (program)
proc main() {           ; topLevelElement (procedureDefinition)
    mov rax, 10      ; topLevelElement (instructionLine)
    ret              ; topLevelElement (instructionLine)
}                      ;
```

## 3. Namespaces

Namespaces provide a mechanism for organizing **AsGex** code and preventing naming conflicts.

*   **`namespaceDefinition`:** Defines a namespace using the `namespace` keyword, followed by an `identifier` for the namespace name, an opening brace `{`, zero or more `topLevelElement` within the namespace, a closing brace `}`, an optional `comment`, and a `lineEnd`. Think of namespaces like folders for your code.

**Example:**

```asgex
namespace Math {        ; namespaceDefinition
    proc add(a: dword, b: dword) -> dword { ; topLevelElement (procedureDefinition) within Math
        mov eax, a
        add eax, b
        ret
    }
}

proc main() {            ; topLevelElement (procedureDefinition)
    call Math::add ; Accessing 'add' from the Math namespace ; instructionLine
}
```

## 4. Concepts

Concepts define requirements on template parameters, enabling more robust and type-safe generic programming in **AsGex**.

*   **`conceptDefinition`:** Defines a concept using the `concept` keyword, followed by an `identifier` for the concept name, optional `<` and `>` enclosing a `templateParameterList`, an optional `whereClause`, an opening brace `{`, zero or more `conceptRequirement`, a closing brace `}`, an optional `comment`, and a `lineEnd`. Concepts allow you to specify what a type must be able to do to be used with a template.
*   **`conceptRequirement`:**  Specifies the requirements within a concept, which can be one of:
    *   `typeRequirement`:  Specifies a nested type or associated type.
    *   `expressionRequirement`: Specifies a required expression that must be valid for the type.
    *   `memberRequirement`: Specifies a required member (field or method) that must exist for the type.
*   **`typeRequirement`:** Specifies that a template parameter must conform to another type or concept, using the `typename` keyword, an `identifier` (the name of the associated type), and an optional `:` followed by a `templateParameter` (a constraint on the associated type).
*   **`expressionRequirement`:** Requires a certain expression involving the template parameter to be valid at compile time, using the `requires` keyword followed by an `expression` and a semicolon `;`. This ensures certain operations are possible with the type.
*   **`memberRequirement`:**  Requires a specific member (either a field or a method) to exist for the type, using the `requires` keyword, an `identifier` (the type), a dot `.`, another `identifier` (the member name), optional parentheses `(` and `)` enclosing an optional `parameterList` (for methods), an optional `->` followed by a `typeReference` (for method return types), and a semicolon `;`.
*   **`whereClause`:** Provides a flexible way to express constraints on template parameters using the `where` keyword followed by an `expression`. This allows for more complex conditions.

**Example:**

```asgex
concept Numeric<T> {   ; conceptDefinition
    typename Result : T; ; conceptRequirement (typeRequirement) - T must have a nested type called Result that is also T
    requires T + T;     ; conceptRequirement (expressionRequirement) - You must be able to add two values of type T
}

concept Addable<T> where T is Numeric<T> { ; conceptDefinition with whereClause
    requires T + T;                      ; conceptRequirement (expressionRequirement)
}

template <typename T>       ; templateDefinition
requires Addable<T>        ; requiresClause - T must satisfy the Addable concept
proc generic_add(a: T, b: T) -> T { ; procedureDefinition
    ; ...
}
```

## 5. Threads

**AsGex** provides language-level support for threads, enabling concurrent programming.

*   **`threadDefinition`:** Defines a thread using the `thread` keyword, followed by an `identifier` for the thread name, optional `<` and `>` enclosing a `templateArgumentList`, optional parentheses `(` and `)` enclosing a `parameterList`, an optional `:` followed by a `typeReference`, an opening brace `{`, zero or more `topLevelElement`, a closing brace `}`, an optional `comment`, and a `lineEnd`.
*   **`parameterList`:**  A list of `parameter` separated by commas `,`. These are the inputs to the thread.
*   **`parameter`:** A single parameter declaration, consisting of an `identifier` for the parameter name, a colon `:`, a `typeReference`, and an optional `=` followed by an `expression` (for a default value).

**Example:**

```asgex
thread MyWorkerThread { ; threadDefinition
    ; ...
}

proc main() {        ; procedureDefinition
    thread my_thread = MyWorkerThread ; ; threadCreation - creates an instance of the thread
    ; ...
    threadjoin my_thread ;            ; threadDirective - waits for the thread to finish
}
```

## 6. Procedures/Functions

Procedures (or functions) encapsulate reusable blocks of code in **AsGex**.

*   **`procedureDefinition`:** Defines a procedure using the `proc` keyword, followed by an `identifier` for the procedure name, optional parentheses `(` and `)` enclosing a `parameterList`, an optional `->` followed by a `typeReference` (specifying the return type), an optional `callingConvention`, an opening brace `{`, zero or more `topLevelElement`, a closing brace `}`, an optional `comment`, and a `lineEnd`.
*   **`callingConvention`:** Specifies the calling convention for the procedure, using the `callingconvention` keyword followed by either `"cdecl"`, `"stdcall"`, `"fastcall"`, or an `identifier` representing a custom calling convention. This determines how arguments are passed and the stack is managed.

**Example:**

```asgex
proc multiply(a: qword, b: qword) -> qword { ; procedureDefinition
    mov rax, a
    mul b
    ret
}
```

## 7. Instructions

Instructions represent the fundamental operations executed by the processor in **AsGex**.

*   **`instruction`:** Represents a single assembly instruction, consisting of an optional `label` followed by a colon `:` and an `instructionBody`. The basic unit of execution.
*   **`instructionBody`:**  Can be one of the `architectureSpecificInstructionBody` options.
*   **`architectureSpecificInstructionBody`:** The core of the instruction, which varies depending on the target architecture:
    *   `x86InstructionBody`  (* @arch: x86, x64, x86-32 *) - Instructions for Intel and AMD processors.
    *   `armInstructionBody`  (* @arch: arm, armv7-a, armv8-a, arm64 *) - Instructions for ARM processors.
    *   `gpuInstructionBody`  (* @arch: gpu, cuda, rocm, ... *) - Instructions for GPUs.
    *   `emptyInstruction`: A "no operation" instruction.
*   **`emptyInstruction`:**  The `nop` instruction. Does nothing.
*   **`instructionPrefix`:** Optional prefixes that modify the instruction's behavior, can be one of:
    *   `x86InstructionPrefix`
    *   `armInstructionPrefix`
    *   `gpuInstructionPrefix`
*   **`shorthandInstruction`:** A simplified syntax for common operations. Makes code more concise.
*   **`mnemonic`:** The instruction's name, consisting of an optional `namespaceQualifier` followed by an `identifier`. The name of the operation (e.g., `mov`, `add`).
*   **`instructionMnemonic`:** The basic instruction mnemonics, can be one of:
    *   `x86InstructionMnemonic`
    *   `armInstructionMnemonic`
    *   `gpuInstructionMnemonic`
*   **`x86InstructionMnemonic`:** An `identifier`.
*   **`armInstructionMnemonic`:** An `identifier`.
*   **`gpuInstructionMnemonic`:** An `identifier`.

**Example:**

```asgex
my_label:           ; instruction (label)
    mov rax, 0x10  ; instruction (instructionBody - x86InstructionBody)
    add rax, rbx   ; instruction (instructionBody - x86InstructionBody)
    jz  end_block  ; instruction (instructionBody - x86InstructionBody)
```

### 7.1 X86 Instructions

*   **`x86InstructionBody`:** Instruction body specific to x86 architectures:
    *   Optional `x86InstructionPrefix`, `x86InstructionMnemonic`, optional `x86OperandList`, optional `comment`, and `lineEnd`.
    *   `x86ShorthandInstruction`, optional `comment`, and `lineEnd`. (* @arch: x86, x64, x86-32 *)
*   **`x86ShorthandInstruction`:** An `x86ModifiableOperand`, a `shorthandOperator`, and an `expression`. For example, `rax += 5` is shorthand for `add rax, 5`.
*   **`x86InstructionPrefix`:** Zero or more `x86Prefix`. These prefixes modify the behavior of the instruction.
*   **`x86Prefix`:** Can be one of:
    *   `x86RepeatPrefix`: For repeating string operations.
    *   `x86SegmentPrefix`: To override the default segment.
    *   `x86AddressPrefix`: To override the default address size.
    *   `x86DataPrefix`: To override the default data size.
    *   `x86VectorPrefix`: For vector instructions.
    *   `x86OtherPrefix`: Other less common prefixes.
*   **`x86RepeatPrefix`:** `rep`, `repe`, `repz`, `repne`, `repnz`, `lock`.
*   **`x86SegmentPrefix`:** `cs`, `ds`, `es`, `fs`, `gs`, `ss`.
*   **`x86AddressPrefix`:** `addr16`, `addr32`, `addr64`.
*   **`x86DataPrefix`:** `byte`, `word`, `dword`, `qword`, `tbyte`.
*   **`x86VectorPrefix`:** `xmmword`, `ymmword`, `zmmword`, `b128`, `b256`, `b512`, `b1024`, `b2048`.
*   **`x86OtherPrefix`:** `bnd`, `notrack`, `gfx`.
*   **`x86OperandList`:**  A list of `x86Operand` separated by commas `,`. The inputs and outputs of the instruction.
*   **`x86Operand`:** An optional `x86OperandSizeOverride`, an optional `x86OperandType`, and an `x86OperandKind`.
*   **`x86OperandSizeOverride`:** `byte`, `word`, `dword`, `qword`, `tbyte`, `oword`, `dqword`, `qqword`, `hqqword`, `vqqword`. Specifies the size of the operand in memory.
*   **`x86OperandType`:** `byte`, `word`, `dword`, `qword`, `xmmword`, `ymmword`, `zmmword`, `b128`, `b256`, `b512`, `b1024`, `b2048`, `ptr`, `far`, `near`, `short`, `tbyte`, `fword`, `signed`, `unsigned`, `threadhandle`. Specifies the data type of the operand.
*   **`x86OperandKind`:** Can be one of:
    *   `immediate`: A constant value.
    *   `x86RegisterOperand`: A CPU register.
    *   `x86MemoryOperand`: A memory location.
    *   `symbolReference`: A named location in memory (label or variable).
*   **`x86ModifiableOperand`:** An optional `x86OperandSizeOverride`, an optional `x86OperandType`, and either an `x86RegisterOperand` or an `x86MemoryOperand`. Operands that can be changed by the instruction.
*   **`x86RegisterOperand`:** An `x86Register`.
*   **`x86MemoryOperand`:** `[` , optional `x86SegmentPrefix` followed by `:`, `x86AddressExpression`, `]`. Accessing data in memory.
*   **`x86AddressExpression`:** `x86AddressBase`, optional `x86AddressOffset`. How the memory address is calculated.
*   **`x86AddressBase`:** Can be one of:
    *   `x86RegisterOperand`: A register holding the base address.
    *   `symbolReference`: A named memory location.
    *   `rel` followed by `symbolReference`:  Address relative to the current instruction pointer.
*   **`x86AddressOffset`:** Can be one of:
    *   `x86AddressDisplacement`: A constant offset.
    *   `x86AddressScaleIndex`: An offset calculated by multiplying a register by a scale factor.
*   **`x86AddressDisplacement`:** Optional `+` or `-`, `x86AddressTerm`, followed by zero or more repetitions of optional `+` or `-` and `x86AddressTerm`. A constant value added to or subtracted from the base.
*   **`x86AddressScaleIndex`:** `+`, `x86RegisterOperand`, `*`, `x86ScaleFactor`. For accessing elements in arrays efficiently.
*   **`x86AddressTerm`:** Can be one of:
    *   `constant`
    *   `x86RegisterOperand`
*   **`x86ScaleFactor`:** `"1"`, `"2"`, `"4"`, `"8"`.
*   **`x86Register`:** Can be one of:
    *   `generalRegister`
    *   `segmentRegister`
    *   `controlRegister`
    *   `debugRegister`
    *   `mmxRegister`
    *   `xmmRegister`
    *   `ymmRegister`
    *   `zmmRegister`
    *   `kRegister`
    *   `bndRegister`

**Example:**

```asgex
    mov rax, [rbx + rcx * 8 + 10] ; x86InstructionBody
                                    ; rax: x86RegisterOperand
                                    ; [rbx + rcx * 8 + 10]: x86MemoryOperand
                                    ; rbx: x86AddressBase (x86RegisterOperand)
                                    ; rcx * 8: x86AddressOffset (x86AddressScaleIndex)
                                    ; 10: x86AddressOffset (x86AddressDisplacement)
    add rsp, 8                ; x86InstructionBody
                                    ; rsp: x86ModifiableOperand (x86RegisterOperand)
    [my_variable] += 5        ; x86ShorthandInstruction
                                    ; [my_variable]: x86ModifiableOperand (x86MemoryOperand)
```

### 7.2 ARM Instructions

*   **`armInstructionBody`:** Instruction body specific to ARM architectures:
    *   Optional `armInstructionPrefix`, `armInstructionMnemonic`, optional `armOperandList`, optional `comment`, and `lineEnd`.
    *   `armShorthandInstruction`, optional `comment`, and `lineEnd`. (* @arch: arm, armv7-a, armv8-a, arm64 *)
*   **`armShorthandInstruction`:** An `armModifiableOperand`, a `shorthandOperator`, and an `armOperand`.
*   **`armInstructionPrefix`:** Zero or more `armConditionCode`, optionally followed by `armHintPrefix`.
*   **`armConditionCode`:** `"eq"`, `"ne"`, `"cs"`, `"cc"`, `"mi"`, `"pl"`, `"vs"`, `"vc"`, `"hi"`, `"ls"`, `"ge"`, `"lt"`, `"gt"`, `"le"`, `"al"`. Conditional execution of instructions.
*   **`armHintPrefix`:** `"wfi"`, `"sev"`, `"yield"`, `"nop"`. Hints to the processor about power saving or scheduling.
*   **`armOperandList`:** A list of `armOperand` separated by commas `,`.
*   **`armOperand`:** An optional `armOperandSizeOverride`, and an `armOperandKind`.
*   **`armOperandSizeOverride`:** `"byte"`, `"word"`, `"dword"`, `"qword"`, `"oword"`, `"dqword"`, `"qqword"`, `"hqqword"`, `"vqqword"`.
*   **`armOperandKind`:** Can be one of:
    *   `immediate`
    *   `armRegister`
    *   `armMemoryOperand`
    *   `symbolReference`
    *   `stringLiteral`
*   **`armModifiableOperand`:** An `armRegister`.
*   **`armRegister`:** `"r0"` ... `"r12"`, `"sp"`, `"lr"`, `"pc"`, `"s0"` ... `"s15"`, `"d0"` ... `"d15"`, `"q0"` ... `"q15"`, `"v0"` ... `"v31"`, `"fpsid"`, `"fpscr"`, `"fpexc"`, `"apsr"`, `"cpsr"`, `"spsr"`. (* @arch: armv7-a, armv8-a, NEON *)
*   **`armMemoryOperand`:** `[`, `armAddress`, `]`.
*   **`armAddress`:** `armAddressBase`, optionally followed by `,` and `armAddressOffset`.
*   **`armAddressBase`:** Can be one of:
    *   `armRegister`
    *   `symbolReference`
    *   `rel` followed by `symbolReference`
*   **`armAddressOffset`:** Can be one of:
    *   `armAddressDisplacement`
    *   `armAddressScaleIndex`
    *   `armShiftedRegister`
*   **`armAddressDisplacement`:** Optional `+` or `-`, `addressTerm`, followed by zero or more repetitions of optional `+` or `-` and `addressTerm`.
*   **`armAddressScaleIndex`:** `armRegister`, `,`, `shiftOperation`.
*   **`armShiftedRegister`:** `armRegister`, `,`, `shiftType`, `expression`. Allows for efficient memory access with shifts.
*   **`shiftType`:** `"lsl"`, `"lsr"`, `"asr"`, `"ror"`, `"rrx"`. Types of bitwise shift operations.
*   **`shiftOperation`:** `shiftType`, space, `expression`.

**Example:**

```asgex
    mov r0, [r1, r2, lsl #2] ; armInstructionBody
                                ; r0: armRegister
                                ; [r1, r2, lsl #2]: armMemoryOperand
                                ; r1: armAddressBase (armRegister)
                                ; r2, lsl #2: armAddressOffset (armShiftedRegister)
    add r3, r4, #5           ; armInstructionBody
                                ; r3: armRegister
                                ; r4: armOperand (armRegister)
                                ; #5: armOperand (immediate)
    r5 += r6                ; armShorthandInstruction
```

### 7.3 GPU Instructions

*   **`gpuInstructionBody`:** Instruction body specific to GPU architectures:
    *   Optional `gpuInstructionPrefix`, `gpuInstructionMnemonic`, optional `gpuOperandList`, optional `comment`, and `lineEnd`.
    *   `gpuShorthandInstruction`, optional `comment`, and `lineEnd`. (* @arch: gpu *)
*   **`gpuShorthandInstruction`:** A `gpuModifiableOperand`, a `shorthandOperator`, and a `gpuOperand`.
*   **`gpuInstructionPrefix`:** Zero or more of `"activemask"` or `"predicated"`. Controls the execution of the instruction based on a mask or predicate.
*   **`gpuOperandList`:** A list of `gpuOperand` separated by commas `,`.
*   **`gpuOperand`:** An optional `gpuOperandSizeOverride`, and a `gpuOperandKind`.
*   **`gpuOperandSizeOverride`:** `"b8"`, `"b16"`, `"b32"`, `"b64"`, `"b128"`, `"b256"`, `"b512"`, `"b1024"`, `"b2048"`, `"f32"`, `"f64"`. Specifies the size of the operand in bits or as a float.
*   **`gpuOperandKind`:** Can be one of:
    *   `immediate`
    *   `gpuRegister`
    *   `gpuMemoryOperand`
    *   `symbolReference`
    *   `stringLiteral`
*   **`gpuModifiableOperand`:** A `gpuRegister`.
*   **`gpuRegister`:** `"r0"` ... `"r15"`, `"f0"` ... `"f15"`, `"p0"` ... `"p7"`, `"v0"` ... `"v15"`, `"%tid.x"`, `"%tid.y"`, `"%tid.z"`, `"%ctaid.x"`, `"%ctaid.y"`, `"%ctaid.z"`, `"%ntid.x"`, `"%ntid.y"`, `"%ntid.z"`. Registers specific to GPU programming, including general-purpose, floating-point, predicate, vector registers, and thread/block identifiers.
*   **`gpuMemoryOperand`:** `[`, `gpuAddress`, `]`.
*   **`gpuAddress`:** Optional `gpuAddressSpace`, `gpuAddressExpression`.
*   **`gpuAddressSpace`:** `"global"`, `"shared"`, `"local"`, `"const"`. Specifies the memory space the operand resides in.
*   **`gpuAddressExpression`:** Can be one of:
    *   `gpuRegister`
    *   `(`, `gpuRegister`, `+`, `immediate`, `)`
    *   `(`, `gpuRegister`, `+`, `gpuRegister`, `)`
    *   `symbolReference`
    *   `expression`

**Example:**

```asgex
    mov.b32 r0, [global r1 + 4] ; gpuInstructionBody
                                 ; r0: gpuRegister
                                 ; [global r1 + 4]: gpuMemoryOperand
                                 ; global: gpuAddressSpace
                                 ; r1 + 4: gpuAddressExpression
    add f0, f1, f2             ; gpuInstructionBody
    v0 += v1                   ; gpuShorthandInstruction
```

## 8. ARM Support

**AsGex** includes support for ARM architectures through a dedicated set of instructions, directives, and macros.

*(Content moved and integrated within the relevant sections, specifically 7.2 Instructions and 9. Macros)*

## 9. Directives

Directives are instructions to the **AsGex** assembler/compiler, controlling aspects of the compilation process or data layout.

*   **`directive`:** Represents an assembler directive, which can be one of the following:
    *   `dataDefinition`: Defines data in memory.
    *   `equateDefinition`: Defines a symbolic constant.
    *   `constDefinition`: Defines a compile-time constant.
    *   `incbinDefinition`: Includes binary data from a file.
    *   `timesDefinition`: Repeats an instruction or data definition.
    *   `segmentDefinition`: Defines a memory segment.
    *   `useDefinition`: Imports symbols from another module.
    *   `typeDefinition`: Creates a type alias.
    *   `mutexDefinition`: Declares a mutex for thread synchronization.
    *   `conditionDefinition`: Declares a condition variable for thread synchronization.
    *   `globalDefinition`: Declares a symbol as globally visible.
    *   `externDefinition`: Declares a symbol that is defined externally.
    *   `alignDefinition`: Aligns code or data in memory.
    *   `sectionDefinition`: Switches to a specific linker section.
    *   `ifDirective`, `elifDirective`, `elseDirective`, `endifDirective`: Conditional compilation.
    *   `ifdefDirective`, `ifndefDirective`, `elifdefDirective`, `elifndefDirective`: Conditional compilation based on symbol definition.
    *   `entryPointDefinition`: Specifies the program's entry point.
    *   `callingConventionDefinition`: Sets the default calling convention.
    *   `acpiDefinition`: Defines ACPI-related data.
    *   `ioDefinition`: Generates I/O instructions.
    *   `structDefinition`: Defines a structure data type.
    *   `cpuDefinition`: Specifies the target CPU.
    *   `bitsDefinition`: Sets the target architecture's bitness.
    *   `stackDefinition`: Reserves space for the stack.
    *   `warningDefinition`: Issues a compiler warning.
    *   `errorDefinition`: Issues a compiler error.
    *   `includeDefinition`: Includes another source file.
    *   `includeOnceDefinition`: Includes a source file only once.
    *   `listDefinition`: Enables listing output.
    *   `nolistDefinition`: Disables listing output.
    *   `debugDefinition`: Includes debugging information.
    *   `orgDefinition`: Sets the program's origin address.
    *   `mapDefinition`: Defines a memory mapping.
    *   `argDefinition`: Declares a command-line argument.
    *   `localDefinition`: Declares a local variable.
    *   `setDefinition`: Sets an assembler variable.
    *   `unsetDefinition`: Unsets an assembler variable.
    *   `assertDefinition`: Performs a compile-time assertion.
    *   `optDefinition`: Enables or disables compiler optimizations.
    *   `evalDefinition`: Evaluates an expression at compile time.
    *   `repDirective`: Repeats an instruction or data definition.
    *   `defaultDirective`: Sets a default value for a symbol.
    *   `exportDefinition`: Makes a symbol visible outside the current module.
    *   `commonDefinition`: Declares a common symbol.
    *   `fileDefinition`: Specifies the source file name.
    *   `lineDefinition`: Specifies the source line number.
    *   `contextDefinition`: Starts a new context for debugging.
    *   `endcontextDefinition`: Ends a context for debugging.
    *   `allocDefinition`: Explicitly allocates memory.
    *   `freeDefinition`: Explicitly frees memory.
    *   `bitfieldDefinition`: Defines a bitfield within a structure.
    *   `gpuDefinition`: Provides GPU-specific directives.
    *   `uefiDefinition`: Provides UEFI-specific directives.
    *   `staticDefinition`: Declares static data.
    *   `dataBlockDefinition`: Defines a block of initialized data.
    *   `gdtDefinition`: Defines the Global Descriptor Table.
    *   `idtDefinition`: Defines the Interrupt Descriptor Table.
    *   `linkerDefinition`: Provides instructions to the linker.
    *   `armDirective`: Provides ARM-specific directives.
*   **`dataDirective`:** `"db"`, `"dw"`, `"dd"`, `"dq"`, `"dt"`, `"do"`, `"ddq"`, `"dqq"`, `"dhq"`, `"dvq"`, `"resb"`, `"resw"`, `"resd"`, `"resq"`, `"rest"`, `"reso"`, `"resdq"`, `"resqq"`, `"reshq"`, `"resvq"`. Defines data of various sizes (byte, word, dword, etc.) or reserves space.
*   **`equateDirective`:** `"equ"`. Assigns a symbolic name to an expression.
*   **`constDirective`:** `"const"`. Declares a constant whose value is known at compile time.
*   **`incbinDirective`:** `"incbin"`. Includes the raw binary contents of a file.
*   **`timesDirective`:** `"times"`. Repeats the following instruction or data definition a specified number of times.
*   **`segmentDirective`:** `"segment"`, `"section"`. Defines or switches to a specific memory segment or linker section.
*   **`useDirective`:** `"use"`. Imports symbols from another module or namespace.
*   **`typeDirective`:** `"type"`. Creates a synonym for an existing type.
*   **`mutexDirective`:** `"mutex"`. Declares a mutual exclusion lock for thread synchronization.
*   **`conditionDirective`:** `"condition"`. Declares a condition variable for thread synchronization.
*   **`globalDirective`:** `"global"`. Makes a symbol visible to other compilation units.
*   **`externDirective`:** `"extern"`. Declares a symbol that is defined in another compilation unit.
*   **`alignDirective`:** `"align"`. Ensures that the next piece of data or code starts at an address that is a multiple of the specified value.
*   **`sectionDirective`:** `"section"`. Specifies the linker section for the following code or data.
*   **`ifDirective`:** `"if"`. Starts a conditional compilation block.
*   **`elifDirective`:** `"elif"`. An "else if" for conditional compilation.
*   **`elseDirective`:** `"else"`. The "else" part of a conditional compilation block.
*   **`endifDirective`:** `"endif"`. Ends a conditional compilation block.
*   **`ifdefDirective`:** `"ifdef"`. Starts a conditional compilation block that is executed if a symbol is defined.
*   **`ifndefDirective`:** `"ifndef"`. Starts a conditional compilation block that is executed if a symbol is not defined.
*   **`elifdefDirective`:** `"elifdef"`. An "else if" for `ifdef`.
*   **`elifndefDirective`:** `"elifndef"`. An "else if" for `ifndef`.
*   **`entryPointDirective`:** `"entry"`. Specifies the starting point of the program's execution.
*   **`callingConventionDirective`:** `"callingconvention"`. Sets the default way functions pass arguments.
*   **`acpiDirective`:** `"acpi"`. Defines data structures related to the Advanced Configuration and Power Interface.
*   **`ioDirective`:** `"io"`. Generates instructions for interacting with I/O ports.
*   **`structDirective`:** `"struct"`. Defines a composite data type.
*   **`cpuDirective`:** `"cpu"`. Specifies the target processor architecture.
*   **`bitsDirective`:** `"bits"`. Sets the word size for the target architecture (16, 32, or 64 bits).
*   **`stackDirective`:** `"stack"`. Reserves space for the program's stack.
*   **`warningDirective`:** `"warning"`. Emits a non-fatal diagnostic message.
*   **`errorDirective`:** `"error"`. Emits a fatal diagnostic message, stopping compilation.
*   **`includeDirective`:** `"include"`. Inserts the contents of another file into the current source file.
*   **`includeOnceDirective`:** `"includeonce"`. Includes a file only if it hasn't been included before.
*   **`listDirective`:** `"list"`. Enables the generation of an assembly listing file.
*   **`nolistDirective`:** `"nolist"`. Disables the generation of an assembly listing file.
```markdown
*   **`debugDirective`:** `"debug"`. Includes debugging information in the output file.
*   **`orgDirective`:** `"org"`. Sets the current location counter to a specific address.
*   **`mapDirective`:** `"map"`. Defines a memory map, associating symbols with addresses.
*   **`argDirective`:** `"arg"`. Declares a command-line argument that the program expects.
*   **`localDirective`:** `"local"`. Declares a variable that is local to the current scope.
*   **`setDirective`:** `"set"`. Assigns a value to an assembler variable.
*   **`unsetDirective`:** `"unset"`. Removes the definition of an assembler variable.
*   **`assertDirective`:** `"assert"`. Checks a condition at compile time and emits an error if it's false.
*   **`optDirective`:** `"opt"`. Enables or disables specific compiler optimizations.
*   **`evalDirective`:** `"eval"`. Evaluates an expression at compile time and inserts the result into the code.
*   **`repDirective`:** `"rep"`. Repeats the following instruction or data definition a specified number of times. Similar to `times`.
*   **`defaultDirective`:** `"default"`. Specifies a default value for a symbol if it's not defined elsewhere.
*   **`exportDirective`:** `"export"`. Makes a symbol visible outside the current module or compilation unit.
*   **`commonDirective`:** `"common"`. Declares a common symbol, which can be defined in multiple compilation units.
*   **`fileDirective`:** `"file"`. Specifies the name of the source file for debugging purposes.
*   **`lineDirective`:** `"line"`. Specifies the line number in the source file for debugging purposes.
*   **`contextDirective`:** `"context"`. Starts a new debugging context.
*   **`endcontextDirective`:** `"endcontext"`. Ends a debugging context.
*   **`allocDirective`:** `"alloc"`. Explicitly allocates a block of memory.
*   **`freeDirective`:** `"free"`. Explicitly frees a block of memory that was previously allocated.
*   **`bitfieldDirective`:** `"bitfield"`. Defines a bitfield within a structure, allowing individual bits to be accessed.
*   **`gpuDirective`:** `"gpu"`. Provides directives specific to GPU programming.
*   **`uefiDirective`:** `"uefi"`. Provides directives specific to UEFI firmware development.
*   **`staticDirective`:** `"static"`. Declares static data, which is initialized only once.
*   **`dataBlockDirective`:** `"datablock"`. Defines a block of initialized data.
*   **`gdtDirective`:** `"gdt"`. Defines entries in the Global Descriptor Table, which is used for memory segmentation.
*   **`idtDirective`:** `"idt"`. Defines entries in the Interrupt Descriptor Table, which is used for handling interrupts and exceptions.
*   **`linkerDirective`:** `"linker"`. Passes instructions directly to the linker.
*   **`armDirective`:** See section 8.16.

*   **`dataDefinition`:** Optional `label` followed by `:`, `dataDirective`, one or more comma-separated `constant` or `stringLiteral`, and `lineEnd`. Example: `my_data: db 10, 20, 30`
*   **`equateDefinition`:** Optional `label` followed by `:`, `equateDirective`, `identifier`, `,`, `expression`, and `lineEnd`. Example: `VALUE equ 100`
*   **`constDefinition`:** Optional `label` followed by `:`, `constDirective`, `identifier`, `=`, `constExpression`, and `lineEnd`. Example: `const MAX_VALUE = 255`
*   **`incbinDefinition`:** `.`, `incbinDirective`, `stringLiteral` (the file path), and `lineEnd`. Example: `.incbin "data.bin"`
*   **`timesDefinition`:** `.`, `timesDirective`, `expression` (the number of repetitions), either `instructionLine` or `dataDefinition`, and `lineEnd`. Example: `.times 5 mov rax, 0`
*   **`segmentDefinition`:** `.`, `segmentDirective`, `identifier` (the segment name), and `lineEnd`. Example: `.segment .data`
*   **`useDefinition`:** `.`, `useDirective`, `namespaceQualifier` (the namespace to import from), and `lineEnd`. Example: `.use MyModule`
*   **`typeDefinition`:** `.`, `typeDirective`, `identifier` (the new type name), `=`, `typeReference` (the existing type), and `lineEnd`. Example: `.type MyInt = dword`
*   **`mutexDefinition`:** `.`, `mutexDirective`, `identifier` (the mutex name), and `lineEnd`. Example: `.mutex my_lock`
*   **`conditionDefinition`:** `.`, `conditionDirective`, `identifier` (the condition variable name), and `lineEnd`. Example: `.condition my_cond`
*   **`globalDefinition`:** `.`, `globalDirective`, `identifier` (the symbol name), and `lineEnd`. Example: `.global my_function`
*   **`externDefinition`:** `.`, `externDirective`, `identifier` (the symbol name), optional `:` followed by `typeReference`, and `lineEnd`. Example: `.extern printf: proc`
*   **`alignDefinition`:** `.`, `alignDirective`, `expression` (the alignment value), and `lineEnd`. Example: `.align 16`
*   **`sectionDefinition`:** `.`, `sectionDirective`, `identifier` (the section name), and `lineEnd`. Example: `.section .text`
*   **`ifDirective`:** `if`, `conditionalExpressionLine`.
*   **`elifDirective`:** `elif`, `conditionalExpressionLine`.
*   **`elseDirective`:** `else`, `lineEnd`.
*   **`endifDirective`:** `endif`, `lineEnd`.
*   **`ifdefDirective`:** `ifdef`, `identifier`, and `lineEnd`.
*   **`ifndefDirective`:** `ifndef`, `identifier`, and `lineEnd`.
*   **`elifdefDirective`:** `elifdef`, `identifier`, and `lineEnd`.
*   **`elifndefDirective`:** `elifndef`, `identifier`, and `lineEnd`.
*   **`entryPointDefinition`:** `.`, `entryPointDirective`, optional `identifier` (the entry point symbol), and `lineEnd`. Example: `.entry main`
*   **`callingConventionDefinition`:** `.`, `callingConventionDirective`, either `"cdecl"`, `"stdcall"`, `"fastcall"`, or `identifier`, and `lineEnd`. Example: `.callingconvention cdecl`
*   **`acpiDefinition`:** `.`, `acpiDirective`, /* specific arguments */ `lineEnd`.
*   **`ioDefinition`:** `.`, `ioDirective`, /* specific arguments */ `lineEnd`.
*   **`structDefinition`:** `"struct"`, `identifier` (the struct name), optional `:` followed by `typeReference` (base struct), `{`, zero or more `structMember`, `}`, optional `comment`, and `lineEnd`. Example:
    ```asgex
    struct Point {
        x: dword ;
        y: dword ;
    }
    ```
*   **`cpuDefinition`:** `.`, `cpuDirective`, `identifier` (the CPU name), and `lineEnd`. Example: `.cpu x64`
*   **`bitsDefinition`:** `.`, `bitsDirective`, either `"16"`, `"32"`, or `"64"`, and `lineEnd`. Example: `.bits 32`
*   **`stackDefinition`:** `.`, `stackDirective`, `expression` (the stack size), and `lineEnd`. Example: `.stack 4096`
*   **`warningDefinition`:** `.`, `warningDirective`, `stringLiteral` (the warning message), and `lineEnd`. Example: `.warning "This is a warning"`
*   **`errorDefinition`:** `.`, `errorDirective`, `stringLiteral` (the error message), and `lineEnd`. Example: `.error "This is an error"`
*   **`includeDefinition`:** `.`, `includeDirective`, `stringLiteral` (the file path), and `lineEnd`. Example: `.include "myfile.asm"`
*   **`includeOnceDefinition`:** `.`, `includeOnceDirective`, `stringLiteral` (the file path), and `lineEnd`. Example: `.includeonce "myfile.asm"`
*   **`listDefinition`:** `.`, `listDirective`, and `lineEnd`.
*   **`nolistDefinition`:** `.`, `nolistDirective`, and `lineEnd`.
*   **`debugDefinition`:** `.`, `debugDirective`, optional `expression`, and `lineEnd`. Example: `.debug`
*   **`orgDefinition`:** `.`, `orgDirective`, `expression` (the address), and `lineEnd`. Example: `.org 0x1000`
*   **`mapDefinition`:** `.`, `mapDirective`, `expression`, `,`, `expression`, and `lineEnd`.
*   **`argDefinition`:** `.`, `argDirective`, `identifier` (the argument name), `:`, `typeReference`, and `lineEnd`. Example: `.arg my_arg: dword`
*   **`localDefinition`:** `.`, `localDirective`, `identifier`, `:`, `typeReference`, optional `=`, `expression`, and `lineEnd`. Example: `.local temp: dword = 0`
*   **`setDefinition`:** `.`, `setDirective`, `identifier`, `=`, `expression`, and `lineEnd`.
*   **`unsetDefinition`:** `.`, `unsetDirective`, `identifier`, and `lineEnd`.
*   **`assertDefinition`:** `.`, `assertDirective`, `expression` (the condition), optional `,`, `stringLiteral` (the error message), and `lineEnd`. Example: `.assert VALUE > 0, "VALUE must be positive"`
*   **`optDefinition`:** `.`, `optDirective`, either `"enable"` or `"disable"`, `,`, `identifier` (the optimization name), and `lineEnd`. Example: `.opt enable, inlining`
*   **`evalDefinition`:** `.`, `evalDirective`, `expression`, and `lineEnd`.
*   **`repDirective`:** `.`, `repDirective`, `expression` (the number of repetitions), either `instructionLine` or `dataDefinition`, and `lineEnd`.
*   **`defaultDirective`:** `.`, `defaultDirective`, `identifier`, `=`, `expression`, and `lineEnd`.
*   **`exportDefinition`:** `.`, `exportDirective`, `identifier`, and `lineEnd`.
*   **`commonDefinition`:** `.`, `commonDirective`, `identifier`, optional `:` followed by `typeReference`, optional `,` followed by `expression` (size), and `lineEnd`.
*   **`fileDefinition`:** `.`, `fileDirective`, `stringLiteral`, and `lineEnd`.
*   **`lineDefinition`:** `.`, `lineDirective`, `number`, optional `,`, `stringLiteral`, and `lineEnd`.
*   **`contextDefinition`:** `.`, `contextDirective`, `stringLiteral`, and `lineEnd`.
*   **`endcontextDefinition`:** `.`, `endcontextDirective`, and `lineEnd`.
*   **`allocDefinition`:** `.`, `allocDirective`, `expression` (size), and `lineEnd`.
*   **`freeDefinition`:** `.`, `freeDirective`, `expression` (address), and `lineEnd`.
*   **`bitfieldDefinition`:** `.`, `bitfieldDirective`, `identifier`, `:`, `number` (size in bits), and `lineEnd`.
*   **`gpuDefinition`:** `.`, `gpuDirective`, /* specific arguments */ `lineEnd`.
*   **`uefiDefinition`:** `.`, `uefiDirective`, /* specific arguments */ `lineEnd`.
*   **`staticDefinition`:** `.`, `staticDirective`, `dataDefinition`, and `lineEnd`.
*   **`dataBlockDefinition`:** `.`, `dataBlockDirective`, `identifier`, /* specific arguments */ `lineEnd`.
*   **`gdtDefinition`:** `.`, `gdtDirective`, /* specific arguments */ `lineEnd`.
*   **`idtDefinition`:** `.`, `idtDirective`, /* specific arguments */ `lineEnd`.
*   **`linkerDefinition`:** `.`, `linkerDirective`, `stringLiteral`, and `lineEnd`.

*   **`conditionalExpressionLine`:** `expression`, `lineEnd`. The condition for `if`, `elif`, `ifdef`, etc.

*   **`conditionalExpression`:** Can be one of:
    *   `logicalOrExpression`: A boolean expression using OR (`||`).
    *   `cpuDetectionExpression`: Checks for specific CPU features.
    *   `armDetectionExpression`: Checks for specific ARM features.
*   **`cpuDetectionExpression`:** Can be one of:
    *   `cpuBrandCheck`:  Checks the CPU brand string.
    *   `cpuIdCheck`: Checks the result of the `CPUID` instruction.
*   **`cpuBrandCheck`:** `"cpu_brand"`, (`"=="`, `"!="`, ), `stringLiteral`. Example: `if cpu_brand == "GenuineIntel"`
*   **`cpuIdCheck`:** `"cpuid."`, `cpuIdLeaf`, [`.`, `cpuIdSubleaf`], `.`, `cpuIdRegister`, (`"=="`, `"!="`, `">"`, `"<"`, `">="`, `"<="`), `constant`. Checks specific bits returned by the `CPUID` instruction. Example: `if cpuid.0x1.edx.25 == 1 ; Check for SSE`
*   **`cpuIdLeaf`:** `hexNumber`. The EAX value passed to `CPUID`.
*   **`cpuIdSubleaf`:** `hexNumber`. The ECX value passed to `CPUID`.
*   **`cpuIdRegister`:** `"eax"`, `"ebx"`, `"ecx"`, `"edx"`. The register to check after executing `CPUID`.

*   **`armDetectionExpression`:** `"arm_version"`, (`"=="`, `"!="`, `">"`, `"<"`, `">="`, `"<="`), `number`. Example: `if arm_version >= 7`

*   **`cpuDetectionDirectiveName`:** `"cpuid"`, `"cpu_brand"`.
*   **`armDetectionDirectiveName`:** `"arm_version"`.

*   **`cpuidDetectionErrorDirective`:** `".cpuid_detection_error"`, (`"error"`, `"warning"`, `"ignore"`). Controls how errors during CPUID detection are handled.
*   **`armVersionDetectionErrorDirective`:** `".arm_version_detection_error"`, (`"error"`, `"warning"`, `"ignore"`). Controls how errors during ARM version detection are handled.

*   **`stringDirective`:**  `"db"`, { `,` , `constant` | `stringLiteral` } ; Defines a string in memory.
*   **`asciiDirective`:** `".ascii"`, { `,` , `stringLiteral` } , `lineEnd`; Defines an ASCII string in memory.
*   **`ascizDirective`:** `".asciz"`, { `,` , `stringLiteral` } , `lineEnd`; Defines a null-terminated ASCII string.

### 8.16 ARM Directives

*   **`armDirective`:**  `.`, `armDirectiveName`, [ `armDirectiveArgumentList` ] ; (* @arch: arm *) Directives specific to the ARM architecture.
*   **`armDirectiveName`:** `"syntax"`, `"arch"`, `"thumb"`, `"arm"`, `"global"`, `"func"`, `"endfunc"`, `"type"`, `"size"`. These control the assembler mode, define symbols, and mark function boundaries.
*   **`armDirectiveArgumentList`:** `armDirectiveArgument`, { `,`, `armDirectiveArgument` }.
*   **`armDirectiveArgument`:** `expression`, `stringLiteral`, `identifier`.

**Examples of ARM Directives:**

```assembly
.arch armv7-a       ; Specify the ARM architecture version
.thumb              ; Switch to Thumb instruction set
.global my_arm_func ; Make my_arm_func globally visible
.type my_arm_func, %function ; Indicate that my_arm_func is a function
my_arm_func:
    ; ARM instructions here
.size my_arm_func, .-my_arm_func ; Set the size of my_arm_func
.endfunc            ; Mark the end of the function
```

## 9. Macros

Macros provide a mechanism for code generation through textual substitution in **AsGex**. They are essentially code templates that can be expanded with different arguments.

*   **`macroDefinition`:** Defines a macro using the `#macro` keyword, followed by an `identifier` for the macro name, optional parentheses `(` and `)` enclosing a `parameterList`, an opening brace `{`, zero or more `macroBodyElement`, a closing brace `}`, an optional `comment`, and a `lineEnd`.
*   **`macroBodyElement`:** The body of the macro, which can contain:
    *   `topLevelElement`
    *   `instructionLine`
    *   `directiveLine`
    *   `macroExpansion`: A call to another macro.
*   **`macroExpansion`:** An `identifier` (the macro name) followed by optional parentheses `(` and `)` enclosing an optional `expressionList`.

**Example:**

```asgex
#macro push_and_increment(reg) { ; macroDefinition
    push reg                   ; macroBodyElement (instructionLine)
    inc reg                    ; macroBodyElement (instructionLine)
}

proc main() {
    mov rcx, 10
    push_and_increment(rcx) ; macroExpansion - Expands to 'push rcx; inc rcx;'
}
```

### 9.1 ARM Macros

*   **`armMacroDefinition`:** Defines an ARM-specific macro using `#arm_macro`, followed by an `identifier`, optional `(`, `armParameterList`, `)`, `{`,  `macroBodyElement` (zero or more), `}`, optional `comment`, and `lineEnd`.
*   **`armParameterList`:** `armParameter`, { `,`, `armParameter` }.
*   **`armParameter`:** An `identifier` representing a parameter to the ARM macro.

**Example of an ARM Macro:**

```assembly
#arm_macro mov_if_equal(dest, src, value) {
    cmp dest, value
    moveq dest, src
}

proc main() {
    mov r0, 5
    mov_if_equal(r1, r2, r0) ; Expands to: cmp r1, r0; moveq r1, r2
}
```

## 10. Modules

Modules provide a higher-level organization for code in **AsGex**, helping to group related functions, data, and other elements.

*   **`moduleDefinition`:** Defines a module using the `%module` keyword, followed by an `identifier` for the module name, an opening brace `{`, zero or more `topLevelElement`, a closing brace `}`, an optional `comment`, and a `lineEnd`.

**Example:**

```asgex
%module MyUtils { ; moduleDefinition
    const BUFFER_SIZE = 256; ; topLevelElement (constDefinition)

    proc clear_buffer(buffer: ptr<byte>) { ; topLevelElement (procedureDefinition)
        ; ...
    }
}
```

## 11. Register Classes

Register classes allow grouping related registers for easier management, particularly in templates or macros that operate on sets of registers.

*   **`registerClassDefinition`:** Defines a register class using the `%regclass` keyword, followed by an `identifier` for the class name, an equals sign `=`, an opening brace `{`, a `registerList`, a closing brace `}`, an optional `comment`, and a `lineEnd`.
*   **`registerList`:** A comma-separated list of `register`.

**Example:**

```asgex
%regclass general_purpose_registers = { ; registerClassDefinition
    rax, rbx, rcx, rdx, rsi, rdi, rsp, rbp, r8, r9, r10, r11, r12, r13, r14, r15
};
```

## 12. Templates

Templates enable generic programming in **AsGex**, allowing you to write code that can work with different types or values without being explicitly specialized for each one.

*   **`templateDefinition`:** Defines a template using the `template` keyword, followed by optional `<` and `>` enclosing a `templateParameterList`, an `identifier` for the template name, optional parentheses `(` and `)` enclosing a `parameterList`, an optional `->` followed by a `typeReference` (for function templates), an optional `requiresClause`, optional `{`, `attributeList`, `}`, `{`, zero or more `templateElement`, `}`, optional `comment`, and `lineEnd`.
*   **`templateParameterList`:**  A list of `templateParameter` separated by commas `,`. These are the placeholders for types or values that will be specified when the template is instantiated.
*   **`templateParameter`:** Can be one of:
    *   `typename` `identifier` [ `requires` `conceptReference` ]: A type parameter, optionally constrained by a concept.
    *   `const` `identifier` `:` `typeReference` [ `=` `constExpression` ]: A non-type parameter (a constant value), with an optional default value.
    *   `...` `identifier`: A variadic template parameter, which can accept any number of arguments.
*   **`requiresClause`:** Specifies constraints on template parameters using the `requires` keyword followed by a `conceptConjunction`.
*   **`conceptConjunction`:** A logical AND (`&&`) of `conceptDisjunction`.
*   **`conceptDisjunction`:** A logical OR (`||`) of `conceptReference`.
*   **`conceptReference`:**  A reference to a `concept`, optionally with template arguments: [ `namespaceQualifier` ], `identifier`, [ `<`, [ `templateArgumentList` ], `>` ].
*   **`templateElement`:** Elements that can appear inside a template definition, including:
    *   `topLevelElement`
    *   `unsafeBlock`: A block of code where certain safety checks are relaxed.
    *   `staticBlock`: A block of static data that is initialized only once.
*   **`unsafeBlock`:** `"unsafe"`, `{`, { `topLevelElement` }, `}`.
*   **`staticBlock`:** `"static"`, `{`, { `dataDefinition` }, `}`.
*   **`templateCall`:** Instantiates a template with specific arguments: [ `namespaceQualifier` ], `identifier`, `<`, [ `templateArgumentList` ], `>`.
*   **`templateArgumentList`:** A list of `templateArgument` separated by commas `,`.
*   **`templateArgument`:** Can be one of:
    *   `typeReference`: A type.
    *   `expression`: A constant expression.
*   **`attributeList`:** `identifier`, { `,`, `identifier` }. A list of attributes (compiler-specific properties).

**Example:**

```asgex
concept Addable<T> {
    requires T + T;
}

template <typename T>       ; templateDefinition
requires Addable<T>        ; requiresClause
proc generic_add(a: T, b: T) -> T { ; procedureDefinition (within a template)
    ; Implementation of addition for type T
}

proc main() {
    mov eax, generic_add<dword>(5, 10) ; templateCall - Instantiating with T = dword
}
```

## 13. Comments

*   **`comment`:**  A single-line comment starting with `;`.
*   **`commentChar`:** Any character allowed in a comment.
*   **`lineEnd`:**  A newline character or the end-of-file marker.

**Example:**

```asgex
; This is a comment
mov rax, 10 ; This is also a comment
```

## 14. Labels

*   **`label`:** An `identifier` followed by a colon (e.g., `my_label:`). Labels are used to mark specific locations in the code, typically for jump instructions or data addresses.

**Example:**

```asgex
my_label:  ; A label
    jmp my_label
```

## 15. Instruction Prefixes (x86)

*   **`instructionPrefix`:** Prefixes that modify instruction behavior on x86 architectures.
*   **`x86RepeatPrefix`:**  Repeat prefixes (e.g., `rep`, `repe`, `repne`). Used with string instructions to repeat the operation multiple times.
*   **`x86SegmentPrefix`:** Segment override prefixes (e.g., `cs`, `ds`, `es`). Force the use of a specific segment register for memory access.
*   **`x86AddressPrefix`:** Address size override prefixes (e.g., `addr16`, `addr32`). Change the default address size (16-bit or 32-bit).
*   **`x86DataPrefix`:** Data size override prefixes (e.g., `byte`, `word`, `dword`). Change the default operand size.
*   **`x86VectorPrefix`:** Vector instruction prefixes (e.g., `xmmword`, `ymmword`). Used with SIMD instructions to specify the size of the vector operands.
*   **`x86OtherPrefix`:** Other x86 specific prefixes (e.g., `bnd`, `notrack`). `bnd` is used with Intel MPX (Memory Protection Extensions) and `notrack` is used to disable certain tracking features.
*   **`armInstructionPrefix`:** Prefixes specific to ARM instructions.
*   **`armConditionCode`:** Conditional execution codes for ARM instructions (e.g., `eq`, `ne`, `cs`). These codes determine whether an instruction is executed based on the status of condition flags.
*   **`armHintPrefix`:** Hint prefixes for ARM instructions (e.g., `wfi`, `sev`). These provide hints to the processor about power management or other optimizations.

**Example of x86 Prefixes:**

```asgex
    rep movsb ; Repeat string move byte instruction
    es mov ax, [bx] ; Use ES segment register
```

**Example of ARM Prefixes:**

```asgex
    moveq r0, #0 ; Move 0 to r0 if the Zero flag is set
    wfi          ; Wait for interrupt
```

## 16. Shorthand Operations

*   **`shorthandOperator`:** Operators used in shorthand instructions (e.g., `+=`, `-=`, `*=`, `/=`, `&=`, `|=`, `^=`, `++`, `--`). These operators provide a more concise way to write common operations.

**Example:**

```asgex
    rax += 5 ; Equivalent to add rax, 5
    [my_var] *= 2 ; Equivalent to mul qword ptr [my_var], 2
```

## 17. Thread Operations

*   **`threadCreation`:** Creates a new thread using the `thread` keyword. `thread` [ `identifier`, `=` ], `templateCall`, [ `(`, [ `expressionList` ] , `)` ].
*   **`expressionList`:** A comma-separated list of `expression`.
*   **`threadDirective`:** Directives for managing threads.
*   **`threadJoinDirective`:** `threadjoin`, `identifier`. Waits for the specified thread to terminate.
*   **`threadTerminateDirective`:** `threadterminate`. Terminates the current thread.
*   **`threadSleepDirective`:** `threadsleep`, `constExpression`. Suspends the current thread for a specified number of milliseconds.
*   **`threadYieldDirective`:** `threadyield`. Yields the processor to another thread.
*   **`threadLocalDirective`:** `threadlocal`, `identifier`, `:`, `typeReference`, [ `=`, `expression` ]. Declares a variable that is local to each thread.

**Example:**

```asgex
thread MyThread {
    ; ... thread code ...
}

proc main() {
    thread t1 = MyThread() ; Create a thread
    ; ...
    threadjoin t1          ; Wait for thread t1 to finish
}
```

## 18. Operands

*   **`operandList`:** A comma-separated list of `operand`.
*   **`operand`:** A single operand for an instruction.
*   **`operandSizeOverride`:** Specifies the size of an operand, overriding the default size.
*   **`operandType`:** Specifies the data type of an operand.
*   **`operandKind`:** The kind of an operand (e.g., `immediate`, `registerOperand`, `memoryOperand`).
*   **`x86OperandList`:** A list of operands specifically for x86 instructions.
*   **`x86Operand`:** A single operand for x86 instructions.
*   **`x86OperandSizeOverride`:** Size overrides for x86 operands.
*   **`x86OperandType`:** Type specifications for x86 operands.
*   **`x86OperandKind`:** Kinds of operands in x86 instructions (e.g., `immediate`, `x86Register`, `x86MemoryOperand`).
*   **`armOperandList`:** A list of operands specifically for ARM instructions.
*   **`armOperand`:** A single operand for ARM instructions.
*   **`armOperandSizeOverride`:** Size overrides for ARM operands.
*   **`armOperandKind`:** Kinds of operands in ARM instructions (e.g., `immediate`, `armRegister`, `armMemoryOperand`).
*   **`gpuOperandList`:** A list of operands specifically for GPU instructions.
*   **`gpuOperand`:** A single operand for GPU instructions.
*   **`gpuOperandSizeOverride`:** Size overrides for GPU operands.
*   **`gpuOperandKind`:** Kinds of operands in GPU instructions.

## 19. Modifiable Operands

*   **`modifiableOperand`:** An operand that can be modified (used in shorthand instructions).
*   **`x86ModifiableOperand`:** A modifiable operand in x86 instructions.
*   **`armModifiableOperand`:** A modifiable operand in ARM instructions.
*   **`gpuModifiableOperand`:** A modifiable operand in GPU instructions.

## 20. Operand Kinds

*   **`immediate`:** A constant value embedded directly in the instruction.
*   **`registerOperand`:** A processor register.
*   **`memoryOperand`:** A memory address, specifying a location in memory where data is stored.
*   **`symbolReference`:**  [ `namespaceQualifier` ], `identifier`. A symbolic name that refers to a memory location, such as a label or variable.

## 21. Registers

*   **`register`:**  A processor register. This can be a general-purpose register, a segment register, a control register, a debug register, or an architecture-specific register (e.g., MMX, XMM, YMM, ZMM for x86, or registers specific to ARM or GPU).
*   **`generalRegister`:** General-purpose registers (e.g., `rax`, `rbx`, `rcx`, `rdx` in x86). These are the most commonly used registers for arithmetic and logical operations.
*   **`segmentRegister`:** Segment registers (e.g., `cs`, `ds`, `es` in x86). Used in segmented memory models to specify different memory segments.
*   **`controlRegister`:** Control registers (e.g., `cr0`, `cr2`, `cr3` in x86). These registers control various operating modes and features of the processor.
*   **`debugRegister`:** Debug registers (e.g., `dr0`, `dr1`, `dr2` in x86). Used for debugging purposes, such as setting breakpoints.
*   **`mmxRegister`:** MMX registers (e.g., `mm0`, `mm1` in x86). Used for early SIMD (Single Instruction, Multiple Data) operations.
*   **`xmmRegister`:** XMM registers (e.g., `xmm0`, `xmm1` in x86). Used for SSE (Streaming SIMD Extensions) instructions, which provide more advanced SIMD capabilities.
*   **`ymmRegister`:** YMM registers (e.g., `ymm0`, `ymm1` in x86). Used for AVX (Advanced Vector Extensions) instructions, which extend SSE with wider vectors.
*   **`zmmRegister`:** ZMM registers (e.g., `zmm0`, `zmm1` in x86). Used for AVX-512 instructions, further extending vector processing capabilities.
*   **`kRegister`:** Mask registers (`k0` ... `k7`) used in AVX-512 for conditional operations on vector elements.
*   **`bndRegister`:** Bound registers (`bnd0` ... `bnd3`) used with Intel MPX for bounds checking.
*   **`armRegister`:** Registers specific to the ARM architecture (e.g., `r0` through `r15`, `sp`, `lr`, `pc`, and others like `apsr`, `cpsr`, `spsr`).
*   **`gpuRegister`:** Registers specific to GPU architectures.

## 22. Constants

*   **`constant`:** A constant value (e.g., a number, a character, a string, an address literal).
*   **`number`:** A decimal number (e.g., `123`, `4567`).
*   **`hexNumber`:** A hexadecimal number (e.g., `0x1A`, `0xFF`).
*   **`binNumber`:** A binary number (e.g., `0b1010`, `0b1111`).
*   **`floatNumber`:** A floating-point number (e.g., `3.14`, `2.718e-5`).
*   **`character`:** A character literal (e.g., `'A'`, `'\n'`).
*   **`escapeSequence`:** An escape sequence within a character or string (e.g., `\n`, `\t`, `\\`, `\'`, `\"`, `\xHH`).
*   **`characterChar`:** A character within a character literal or string.
*   **`addressLiteral`:** An address specified as a hexadecimal number (e.g., `$0x1000`).
*   **`cpuIdLiteral`**: The literal `"cpuid"` used in CPU feature detection.
*   **`armVersionLiteral`**: The literal `"arm_version"` used in ARM version detection.
*   **`constExpression`:** An expression that can be evaluated at compile time.
*   **`constValue`:** A value that is known at compile time.
*   **`constSymbol`:** A named constant.

## 23. Expressions

*   **`expression`:** An expression that can be evaluated at compile time or runtime.
*   **`conditionalExpression`:** A conditional expression using the ternary operator (`? :`).
*   **`logicalOrExpression`:** A logical OR expression (`||`).
*   **`logicalAndExpression`:** A logical AND expression (`&&`).
*   **`bitwiseOrExpression`:** A bitwise OR expression (`|`).
*   **`bitwiseXorExpression`:** A bitwise XOR expression (`^`).
*   **`bitwiseAndExpression`:** A bitwise AND expression (`&`).
*   **`shiftExpression`:** A bit shift expression (`<<` or `>>`).
```markdown
*   **`additiveExpression`:** An addition or subtraction expression (`+` or `-`).
*   **`multiplicativeExpression`:** A multiplication, division, or modulo expression (`*`, `/`, or `%`).
*   **`unaryExpression`:** A unary expression (e.g., `+`, `-`, `~`, `!`, `sizeof`, `alignof`, type conversions).
*   **`typeConversion`:** A type conversion (e.g., `byte`, `word`, `dword`, `signed`, `unsigned`).
*   **`sizeOfExpression`:**  The `sizeof` operator, which returns the size of a type or expression in bytes.
*   **`alignOfExpression`:** The `alignof` operator, which returns the alignment requirement of a type.
*   **`cpuIdOperand`**: The `cpuid` literal used as an operand.
*   **`armVersionOperand`**: The `arm_version` literal used as an operand.

**Example of Expressions:**

```asgex
mov rax, 10 + 5 * 2  ; additiveExpression and multiplicativeExpression
mov rbx, ~rax       ; unaryExpression (bitwise NOT)
mov rcx, sizeof(dword) ; sizeOfExpression
```

## 24. Memory Addresses

*   **`memoryOperand`:** Specifies a memory address using square brackets `[]`.
*   **`architectureSpecificMemoryOperand`:**
    *   `x86MemoryOperand`
    *   `armMemoryOperand`
    *   `gpuMemoryOperand`
*   **`addressBase`:** The base register or symbol for the address.
*   **`addressOffset`:** An offset from the base address.
*   **`addressDisplacement`:**  An offset using a constant or register.
*   **`addressScaleIndex`:** An offset using a scaled register (e.g., `[ebx + ecx*4]`).
*   **`addressTerm`:** A component of an address offset (either a constant or a register).
*   **`scaleFactor`:** The scaling factor (1, 2, 4, or 8) for an index register.
*   **`x86AddressBase`:** The base for an x86 memory address, which can be an `x86RegisterOperand`, a `symbolReference`, or a relative reference (`rel` combined with a `symbolReference`).
*   **`x86AddressOffset`:** The offset in an x86 memory address, specified as an `x86AddressDisplacement` or an `x86AddressScaleIndex`.
*   **`x86AddressDisplacement`:** Specifies an offset in an x86 memory address using a combination of constants and `x86RegisterOperands`.
*   **`x86AddressScaleIndex`:** Specifies an offset in an x86 memory address that includes a scaled `x86RegisterOperand`.
*   **`x86AddressTerm`:** A component of an `x86AddressDisplacement`, which can be a `constant` or an `x86RegisterOperand`.
*   **`x86ScaleFactor`:** The scaling factor for an x86 address index, with possible values of "1", "2", "4", or "8".
*   **`armAddressBase`:** The base for an ARM memory address, which can be an `armRegister`, a `symbolReference`, or a relative reference (`rel` combined with a `symbolReference`).
*   **`armAddressOffset`:** The offset in an ARM memory address, specified as an `armAddressDisplacement`, an `armAddressScaleIndex`, or an `armShiftedRegister`.
*   **`armAddressDisplacement`:** Specifies an offset in an ARM memory address using a combination of constants and `addressTerms`.
*   **`armAddressScaleIndex`:** Specifies an offset in an ARM memory address that includes an `armRegister` and a `shiftOperation`.
*   **`armShiftedRegister`:** Specifies a shifted register operand in ARM instructions, including an `armRegister`, a `shiftType`, and an `expression`.

**Example of Memory Addresses (x86):**

```asgex
mov rax, [rbx]            ; Base: rbx
mov rcx, [my_data + 10]   ; Base: my_data, Offset: 10 (addressDisplacement)
mov rdx, [rsi + rdi*4]    ; Base: rsi, Offset: rdi*4 (addressScaleIndex)
```

**Example of Memory Addresses (ARM):**

```asgex
ldr r0, [r1]          ; Base: r1
ldr r2, [r3, #4]      ; Base: r3, Offset: 4 (armAddressDisplacement)
ldr r4, [r5, r6, lsl #2] ; Base: r5, Offset: r6 shifted left by 2 (armShiftedRegister)
```

## 25. String Literals

*   **`stringLiteral`:** A sequence of characters enclosed in double quotes (e.g., `"Hello, world!"`).
*   **`stringCharacter`:** A character within a string literal, which can be either an `escapeSequence` or any character except a double quote or backslash unless escaped.

**Example:**

```asgex
my_string: db "Hello, world!\n", 0 ; String literal
```

## 26. Supplementary Definitions

*   **`namespaceQualifier`:** `identifier`, { `::`, `identifier` }. Used to specify a namespace, for example `MyNamespace::MySubNamespace`.

## 27. Architecture-Specific Blocks

*   **`architectureSpecificBlock`:**  `architectureSpecificTag`, `{`, { `architectureSpecificElement` }, `}`, [ `comment` ], `lineEnd` ; A block of code that is only compiled for a specific target architecture.
*   **`architectureSpecificTag`:** `"@x86"`, `"@x64"`, `"@x86-32"`, `"@arm"`, `"@arm64"`, `"@armv7-a"`, `"@armv8-a"`, `"@gpu"`, `"@cuda"`, `"@rocm"`, `identifier` ; Identifies the target architecture for the block.
*   **`architectureSpecificElement`:** Elements that can appear within an architecture-specific block, such as `instructionLine`, `directiveLine`, etc.

**Example:**

```asgex
@x86 {
    mov eax, 1 ; This code will only be compiled for x86
}

@arm {
    mov r0, #1 ; This code will only be compiled for ARM
}
```

## 28. Enums

*   **`enumDefinition`:** `"enum"`, `identifier`, [ `:`, `typeReference` ], `{`, `enumItemList`, `}`, [ `comment` ], `lineEnd` ; Defines an enumeration type.
*   **`enumItemList`:** `enumItem`, { `,`, `enumItem` }, [ `,` ] ; A list of enum items.
*   **`enumItem`:** `identifier`, [ `=`, `constExpression` ] ; An item in the enumeration, with an optional value.

**Example:**

```asgex
enum Color : byte { ; Defines an enum called Color, where each item is a byte
    Red = 0,
    Green = 1,
    Blue = 2
}
```

## 29. Structs

*   **`structDefinition`:** `"struct"`, `identifier`, [ `:`, `typeReference` ], `{`, { `structMember` }, `}`, [ `comment` ], `lineEnd` ; Defines a structure type.
*   **`structMember`:** `typeReference`, `identifier`, [ `=`, `expression` ], `;`, [ `comment` ], `lineEnd` ; A member of the structure (a field).
*   **`typeReference`:**  [ `namespaceQualifier` ], `identifier` ; A reference to a type (e.g., `dword`, `ptr<byte>`, a user-defined type).

**Example:**

```asgex
struct Point { ; Defines a struct called Point
    x: dword ;
    y: dword ;
}

struct Rect : Point {
    width: dword ;
    height: dword ;
}
```

## Conclusion

This document has provided a comprehensive overview of the **AsGex** Assembly Language grammar. By understanding these rules and definitions, you can effectively write and understand **AsGex** code, leveraging its features for **zero-overhead abstractions, memory-safe templates, and compile-time metaprogramming** to create efficient and powerful low-level programs. Remember that this is a reference guide; don't hesitate to experiment and explore the language's capabilities. As you become more familiar with **AsGex**, you'll discover how its unique design can be a valuable tool for your development projects.
```
