## AsGex Assembly Language Documentation

This document provides a comprehensive overview of the syntax and semantics of the **AsGex** Assembly Language. **AsGex** is designed with a focus on **zero-overhead abstractions**, **memory-safe templates**, and **compile-time metaprogramming** capabilities, targeting a variety of architectures including X64, 32-bit, ARM, GPU, and UEFI/BIOS environments.

**1. Introduction**

**AsGex** aims to provide a powerful and expressive assembly-level language that transcends the limitations of traditional assembly. **Its core selling points are the ability to write low-level code with high-level abstractions that incur no runtime cost, and the power of compile-time metaprogramming for code generation and optimization.** It incorporates modern language concepts like namespaces, templates, concepts, and modules, allowing for more structured and maintainable code while retaining fine-grained control over hardware. The language emphasizes compile-time operations and incorporates features aimed at improving memory safety where possible. **AsGex's multi-architecture focus makes it suitable for a wide range of embedded, system-level, and high-performance computing tasks.**

**2. Program Structure**

An **AsGex** program consists of a sequence of top-level elements, processed sequentially by the compiler.

*   **`program`:** The top-level rule representing the entire program.
*   **`topLevelElement`:**  Represents the different kinds of elements that can appear at the top level of a program (e.g., instructions, directives, definitions).

**Example:**

```asgex
; A simple AsGex program
proc main() {
    mov rax, 10
    ret
}
```

**3. Namespaces**

Namespaces provide a mechanism for organizing **AsGex** code and preventing naming conflicts.

*   **`namespaceDefinition`:** Defines a namespace using the `namespace` keyword.

**Example:**

```asgex
namespace Math {
    proc add(a: dword, b: dword) -> dword {
        mov eax, a
        add eax, b
        ret
    }
}

proc main() {
    call Math::add ; Accessing 'add' from the Math namespace
}
```

**4. Concepts**

Concepts define requirements on template parameters, enabling more robust and type-safe generic programming in **AsGex**.

*   **`conceptDefinition`:** Defines a concept using the `concept` keyword.
*   **`conceptRequirement`:**  Specifies the requirements within a concept (either `typeRequirement` or `expressionRequirement`).
*   **`typeRequirement`:** Specifies that a template parameter must conform to another type or concept.
*   **`expressionRequirement`:** Requires a certain expression involving the template parameter to be valid at compile time.
*   **`whereClause`:** Provides a flexible way to express constraints on template parameters using boolean expressions.

**Example:**

```asgex
concept Addable<T> where T is Numeric {
    typename Result : T; 
    requires T + T;    
}

template <typename T>
requires Addable<T> 
proc generic_add(a: T, b: T) -> T {
    ; ...
}
```

**5. Threads**

**AsGex** provides language-level support for threads, enabling concurrent programming.

*   **`threadDefinition`:** Defines a thread using the `thread` keyword.
*   **`parameterList`:**  A list of parameters for a thread (similar to procedure parameters).
*   **`parameter`:** A single parameter declaration (identifier and type).

**Example:**

```asgex
thread MyWorkerThread {
    ; ...
}

proc main() {
    thread my_thread = MyWorkerThread ;
    ; ...
    threadjoin my_thread ;
}
```

**6. Procedures/Functions**

Procedures (or functions) encapsulate reusable blocks of code in **AsGex**.

*   **`procedureDefinition`:** Defines a procedure using the `proc` keyword.
*   **`callingConventionSpecifier`:** Specifies the calling convention for the procedure (e.g., `cdecl`, `stdcall`, `fastcall`).

**Example:**

```asgex
proc multiply(a: qword, b: qword) -> qword {
    mov rax, a
    mul b
    ret
}
```

**7. Instructions**

Instructions represent the fundamental operations executed by the processor in **AsGex**.

*   **`instruction`:** Represents a single assembly instruction.
*   **`label`:** An optional label to mark the instruction's location.
*   **`instructionPrefix`:** Optional prefixes that modify the instruction's behavior (e.g., `rep`, segment overrides).
*   **`instructionBody`:** The core of the instruction: either a `mnemonic` with operands or a `shorthandInstruction`.
*   **`shorthandInstruction`:** A simplified syntax for common operations (e.g., `rax += 5`).
*   **`mnemonic`:** The instruction's name (e.g., `mov`, `add`, `jmp`).
*   **`namespaceQualifier`:**  Used to specify the namespace of an instruction (if needed).
*   **`instructionMnemonic`:** The basic instruction mnemonics (e.g., `mov`, `add`, `sub`, `jmp`, etc.).
*   **`operandList`:** A list of operands for the instruction.

**Example:**

```asgex
my_label:
    mov rax, 0x10 
    add rax, rbx  
    jz  end_block 
```

**8. ARM Support (Optional)**

**AsGex** includes optional support for ARM architectures through a dedicated set of instructions, directives, and macros.

*   **`armInstruction`:** Represents an ARM instruction.
*   **`armInstructionPrefix`:**  Optional prefixes for ARM instructions.
*   **`armInstructionBody`:** The core of an ARM instruction.
*   **`armShorthandInstruction`:** Shorthand syntax for ARM instructions.
*   **`armMnemonic`:** The name of an ARM instruction.
*   **`armInstructionMnemonic`:** The basic ARM instruction mnemonics.
*   **`armDirective`:** Directives specific to ARM.
*   **`armDirectiveName`:** Names of ARM-specific directives.
*   **`armDirectiveArguments`:** Arguments for ARM directives.
*   **`armMacroDefinition`:** Defines an ARM-specific macro.
*   **`armParameterList`:**  A list of parameters for an ARM macro.
*   **`armParameter`:** A single parameter declaration for an ARM macro.
*   **`armOperandList`:** A list of operands for ARM instructions.
*   **`armOperand`:** A single operand for an ARM instruction.
*   **`armOperandSizeOverride`:** Size overrides for ARM operands.
*   **`armOperandKind`:** The type of an ARM operand (e.g., immediate, register, memory).
*   **`armModifiableOperand`:**  An operand that can be modified in a shorthand instruction.
*   **`armRegister`:** ARM registers (e.g., `r0`, `r1`, `sp`, `lr`, `pc`).
*   **`armMemoryOperand`:**  ARM memory addressing modes.
*   **`armAddressBase`:** The base register for an ARM memory address.
*   **`armAddressOffset`:** The offset from the base register.
*   **`armAddressDisplacement`:**  An offset using a constant or register.
*   **`armAddressScaleIndex`:** An offset using a scaled register (e.g., `r1*4`).

**9. Directives**

Directives are instructions to the **AsGex** assembler/compiler, controlling aspects of the compilation process or data layout.

*   **`directive`:** Represents an assembler directive.
*   **`directiveName`:** The name of the directive (see the list below).
*   **`directiveArguments`:** Arguments for the directive.
*   **`dataDirective`:** Directives for defining data (e.g., `db`, `dw`, `dd`, `dq`, `string`).
*   **`equateDirective`:** Assigns a symbolic name to an expression (e.g., `.equ`).
*   **`constDirective`:** Declares a compile-time constant.
*   **`incbinDirective`:** Includes binary data from a file.
*   **`timesDirective`:** Repeats an instruction or data definition multiple times.
*   **`segmentDirective`:** Defines a memory segment.
*   **`useDirective`:** Imports symbols from another module or namespace.
*   **`typeDirective`:** Creates a named alias for a type.
*   **`mutexDirective`:** Declares a mutex for thread synchronization.
*   **`conditionDirective`:** Declares a condition variable for thread synchronization.
*   **`globalDirective`:** Declares symbols as globally visible.
*   **`externDirective`:** Declares symbols that are defined externally.
*   **`alignDirective`:** Aligns code or data to a specific memory boundary.
*   **`sectionDirective`:**  Switches to a specific linker section.
*   **`ifDirective`:**  Conditional compilation directives (`if`, `elif`, `else`, `endif`).
*   **`entryPointDirective`:**  Specifies the program's entry point.
*   **`callingConventionDirective`:** Sets the default calling convention.
*   **`acpiDirective`:** Defines ACPI-related data structures.
*   **`ioDirective`:** Generates I/O instructions.
*   **`structDirective`:** Defines a structure data type.
*   **`cpuDirective`:** Specifies the target CPU.
*   **`bitsDirective`:** Sets the target architecture's bitness.
*   **`stackDirective`:** Reserves space for the stack.
*   **`warningDirective`:** Issues a compiler warning.
*   **`errorDirective`:** Issues a compiler error.
*   **`includeDirective`:** Includes another source file.
*   **`includeOnceDirective`:** Includes a source file only once.
*   **`listDirective`:** Enables the generation of a listing file.
*   **`nolistDirective`:** Disables the generation of a listing file.
*   **`debugDirective`:** Embeds debugging information.
*   **`orgDirective`:** Sets the origin address.
*   **`mapDirective`:** Defines a memory mapping.
*   **`argDirective`:** Declares a command-line argument.
*   **`localDirective`:** Declares a local variable.
*   **`setDirective`:** Sets an environment variable or compiler flag.
*   **`unsetDirective`:** Unsets an environment variable or compiler flag.
*   **`assertDirective`:** Performs a compile-time assertion.
*   **`optDirective`:** Enables or disables compiler optimizations.
*   **`evalDirective`:** Evaluates an expression at compile time.
*   **`repDirective`:** Repeats an instruction or data definition (similar to `times`).
*   **`defaultDirective`:** Sets a default value.
*   **`ifdefDirective`:** Conditional compilation based on identifier definition.
*   **`ifndefDirective`:** Conditional compilation based on identifier not being defined.
*   **`elifdefDirective`:** `else if` for `ifdef`.
*   **`elifndefDirective`:** `else if` for `ifndef`.
*   **`exportDirective`:** Makes a symbol visible outside the current module.
*   **`commonDirective`:** Declares a common symbol.
*   **`fileDirective`:** Specifies the source file name (for debugging).
*   **`lineDirective`:** Specifies the source line number (for debugging).
*   **`contextDirective`:**  Starts a new context (for debugging).
*   **`endcontextDirective`:** Ends a context (for debugging).
*   **`allocDirective`:** Explicitly allocates memory.
*   **`freeDirective`:** Frees allocated memory.
*   **`bitfieldDirective`:** Defines a bitfield.
*   **`gpuDirective`:**  Provides GPU-specific directives.
*   **`uefiDirective`:** Provides UEFI-specific directives.
*   **`staticDirective`:** Declares static data.
*   **`dataBlockDirective`:** Defines a block of initialized data.
*   **`gdtDirective`:** Defines the Global Descriptor Table.
*   **`idtDirective`:** Defines the Interrupt Descriptor Table.
*   **`linkerDirective`:** Provides instructions to the linker.

**10. Macros**

Macros provide a mechanism for code generation through textual substitution in **AsGex**.

*   **`macroDefinition`:** Defines a macro using the `#macro` keyword.

**Example:**

```asgex
#macro push_and_increment(reg) {
    push reg
    inc reg
}

proc main() {
    mov rcx, 10
    push_and_increment(rcx) ; Expands to 'push rcx; inc rcx;'
}
```

**11. Modules**

Modules provide a higher-level organization for code in **AsGex**.

*   **`moduleDefinition`:** Defines a module using the `%module` keyword.

**Example:**

```asgex
%module MyUtils {
    const BUFFER_SIZE = 256;

    proc clear_buffer(buffer: ptr<byte>) {
        ; ...
    }
}
```

**12. Register Classes**

Register classes allow grouping related registers for easier management.

*   **`registerClassDefinition`:** Defines a register class using the `%regclass` keyword.
*   **`registerList`:** A comma-separated list of registers.

**Example:**

```asgex
%regclass general_purpose_registers = { rax, rbx, rcx, rdx, rsi, rdi, rsp, rbp, r8, r9, r10, r11, r12, r13, r14, r15 };
```

**13. Templates**

Templates enable generic programming in **AsGex**.

*   **`templateDefinition`:** Defines a template using the `template` keyword.
*   **`templateParameterList`:**  A list of template parameters.
*   **`templateParameter`:** A single template parameter declaration (can be `typename`, `const`, or variadic).
*   **`requiresClause`:** Specifies constraints on template parameters using concepts.
*   **`conceptConjunction`:** A conjunction (logical AND) of concept disjunctions.
*   **`conceptDisjunction`:** A disjunction (logical OR) of concept references.
*   **`conceptReference`:**  A reference to a concept, optionally with template arguments.
*   **`templateElement`:** Elements that can appear inside a template definition.
*   **`unsafeBlock`:** Marks a block of code as potentially unsafe (relaxing some compiler checks).
*   **`staticBlock`:** Defines a block of static data within a template.
*   **`templateCall`:** Instantiates a template with specific arguments.
*   **`templateArgumentList`:** A list of template arguments.
*   **`templateArgument`:** A single template argument (either a type or a constant expression).

**Example:**

```asgex
template <typename T>
requires Addable<T>
proc generic_add(a: T, b: T) -> T {
    ; ... 
}
```

**14. Comments**

*   **`comment`:**  A single-line comment starting with `;`.
*   **`commentChar`:** Any character allowed in a comment.
*   **`lineEnd`:**  A newline character or the end-of-file marker.

**15. Labels**

*   **`label`:** An identifier followed by a colon (e.g., `my_label:`).

**16. Instruction Prefixes**

*   **`instructionPrefix`:** Prefixes that modify instruction behavior.
*   **`repeatPrefix`:**  Repeat prefixes (e.g., `rep`, `repe`, `repne`).
*   **`segmentPrefix`:** Segment override prefixes (e.g., `cs`, `ds`, `es`).
*   **`addressPrefix`:** Address size override prefixes (e.g., `addr16`, `addr32`).
*   **`dataPrefix`:** Data size override prefixes (e.g., `byte`, `word`, `dword`).
*   **`vectorPrefix`:** Vector instruction prefixes (e.g., `xmmword`, `ymmword`).
*   **`otherPrefix`:** Other prefixes (e.g., `bnd`, `notrack`).

**17. Shorthand Operations**

*   **`shorthandOperator`:** Operators used in shorthand instructions (e.g., `+=`, `-=`, `*=`, `/=`, `&=`, `|=`, `^=`, `++`, `--`).

**18. Thread Operations**

*   **`threadCreation`:** Creates a new thread.
*   **`expressionList`:** A comma-separated list of expressions.
*   **`threadDirective`:** Directives for managing threads (e.g., `threadjoin`, `threadterminate`, `threadsleep`, `threadyield`).
*   **`threadJoinDirective`:** Waits for a thread to terminate.
*   **`threadTerminateDirective`:** Terminates a thread.
*   **`threadSleepDirective`:** Pauses a thread for a specified duration.
*   **`threadYieldDirective`:** Yields the processor to another thread.
*   **`threadLocalDirective`:**  Declares thread-local data.

**19. Operands**

*   **`operandList`:** A comma-separated list of operands.
*   **`operand`:** A single operand for an instruction.
*   **`operandSizeOverride`:** Specifies the size of an operand.
*   **`operandType`:** Specifies the type of an operand.
*   **`operandKind`:** The kind of an operand (e.g., `immediate`, `registerOperand`, `memoryOperand`).

**20. Modifiable Operands**

*   **`modifiableOperand`:** An operand that can be modified (used in shorthand instructions).

**21. Operand Kinds**

*   **`immediate`:** A constant value.
*   **`registerOperand`:** A register.
*   **`memoryOperand`:** A memory address.

**22. Registers**

*   **`register`:** A general-purpose, segment, control, debug, MMX, XMM, YMM, or ZMM register.
*   **`generalRegister`:** General-purpose registers (e.g., `rax`, `rbx`, `rcx`, `rdx`).
*   **`segmentRegister`:** Segment registers (e.g., `cs`, `ds`, `es`).
*   **`controlRegister`:** Control registers (e.g., `cr0`, `cr2`, `cr3`).
*   **`debugRegister`:** Debug registers (e.g., `dr0`, `dr1`, `dr2`).
*   **`mmxRegister`:** MMX registers (e.g., `mm0`, `mm1`).
*   **`xmmRegister`:** XMM registers (e.g., `xmm0`, `xmm1`).
*   **`ymmRegister`:** YMM registers (e.g., `ymm0`, `ymm1`).
*   **`zmmRegister`:** ZMM registers (e.g., `zmm0`, `zmm1`).

**23. Constants**

*   **`constant`:** A constant value (e.g., a number, a character, a string, an address literal).
*   **`number`:** A decimal number.
*   **`hexNumber`:** A hexadecimal number (e.g., `0x1A`, `0xFF`).
*   **`binNumber`:** A binary number (e.g., `0b1010`, `0b1111`).
*   **`floatNumber`:** A floating-point number.
*   **`character`:** A character literal (e.g., `'A'`, `'\n'`).
*   **`escapeSequence`:** An escape sequence within a character or string (e.g., `\n`, `\t`, `\\`, `\'`, `\"`, `\xHH`).
*   **`characterChar`:** A character within a character literal or string.
*   **`addressLiteral`:** An address specified as a hexadecimal number (e.g., `$0x1000`).

**24. Expressions**

*   **`expression`:** An expression that can be evaluated at compile time or runtime.
*   **`conditionalExpression`:** A conditional expression using the ternary operator (`? :`).
*   **`logicalOrExpression`:** A logical OR expression (`||`).
*   **`logicalAndExpression`:** A logical AND expression (`&&`).
*   **`bitwiseOrExpression`:** A bitwise OR expression (`|`).
*   **`bitwiseXorExpression`:** A bitwise XOR expression (`^`).
*   **`bitwiseAndExpression`:** A bitwise AND expression (`&`).
*   **`shiftExpression`:** A bit shift expression (`<<` or `>>`).
*   **`additiveExpression`:** An addition or subtraction expression (`+` or `-`).
*   **`multiplicativeExpression`:** A multiplication, division, or modulo expression (`*`, `/`, or `%`).
*   **`unaryExpression`:** A unary expression (e.g., `+`, `-`, `~`, `!`, `sizeof`, `alignof`, type conversions).
*   **`typeConversion`:** A type conversion (e.g., `byte`, `word`, `dword`, `signed`, `unsigned`).
*   **`sizeOfExpression`:**  The `sizeof` operator.
*   **`alignOfExpression`:** The `alignof` operator.

**25. Memory Addresses**

*   **`memoryAddress`:** Specifies a memory address using square brackets `[]`.
*   **`addressBase`:** The base register or symbol for the address.
*   **`addressOffset`:** An offset from the base address.
*   **`addressDisplacement`:**  An offset using a constant or register.
*   **`addressScaleIndex`:** An offset using a scaled register (e.g., `[ebx + ecx*4]`).
*   **`addressTerm`:** A component of an address offset (either a constant or a register).
*   **`scaleFactor`:** The scaling factor (1, 2, 4, or 8) for an index register.

**26. String Literals**

*   **`stringLiteral`:** A sequence of characters enclosed in double quotes (e.g., `"Hello, world!"`).
*   **`stringChar`:** A character within a string literal.

**27. Lexical Tokens**

*   **`identifier`:** A name for a variable, label, function, etc.
*   **`digit`:** A decimal digit (0-9).
*   **`hexDigit`:** A hexadecimal digit (0-9, A-F).
*   **`binDigit`:** A binary digit (0 or 1).
*   **`eof`:** The end-of-file marker.



 
