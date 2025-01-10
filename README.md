
## Documentation for the AsGex Assembly Language

This document provides a comprehensive overview of the syntax and semantics of the **AsGex** Assembly Language, as defined by the provided Enhanced Backus-Naur Form (EBNF) grammar. **AsGex** is designed with a focus on **zero-overhead abstractions**, **memory-safe templates**, and **compile-time metaprogramming** capabilities, targeting a variety of architectures including X64, 32-bit, GPU, and UEFI/BIOS environments.

**1. Introduction**

**AsGex** aims to provide a powerful and expressive assembly-level language that transcends the limitations of traditional assembly. **Its core selling points are the ability to write low-level code with high-level abstractions that incur no runtime cost, and the power of compile-time metaprogramming for code generation and optimization.**  It incorporates modern language concepts like namespaces, templates, concepts, and modules, allowing for more structured and maintainable code while retaining fine-grained control over hardware. The language emphasizes compile-time operations and incorporates features aimed at improving memory safety where possible. **AsGex's multi-architecture focus makes it suitable for a wide range of embedded, system-level, and high-performance computing tasks.**

**2. Program Structure**

An **AsGex** program consists of a sequence of top-level elements, processed sequentially by the compiler.

```ebnf
program = { topLevelElement }, eof ;
topLevelElement = instruction | directive | macroDefinition | templateDefinition | moduleDefinition | registerClassDefinition | threadDefinition | enumDefinition | structDefinition | threadLocalDirective | namespaceDefinition | conceptDefinition | procedureDefinition ;
```

The program terminates at the end-of-file (eof) marker.

**3. Namespaces**

Namespaces provide a mechanism for organizing **AsGex** code and preventing naming conflicts.

```ebnf
namespaceDefinition = "namespace", identifier, "{", { topLevelElement }, "}" ;
```

*   A namespace is defined using the `namespace` keyword, followed by an identifier (the namespace name), and a block `{}` containing top-level elements that belong to this namespace.
*   To access elements within a namespace, use the `::` scope resolution operator (e.g., `MyNamespace::myFunction`).

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
    mov ecx, 10
    mov edx, 20
    call Math::add ; Accessing 'add' from the Math namespace
    ; ...
}
```

**4. Concepts**

Concepts define requirements on template parameters, enabling more robust and type-safe generic programming in **AsGex**.

```ebnf
conceptDefinition = "%concept", identifier, [ "<", templateParameterList, ">" ], "{", { conceptRequirement }, "}" ;
conceptRequirement = typeRequirement | expressionRequirement ;
typeRequirement = "typename", identifier, ":", templateParameter ;
expressionRequirement = "requires", expression, ";" ;
```

*   Concepts are introduced with the `%concept` keyword, followed by an identifier (the concept name) and optional template parameters.
*   The concept body `{}` contains requirements that a type must satisfy to be considered a model of the concept.
*   **Type Requirements:** Specify that a template parameter must be a type that itself conforms to another template parameter (effectively a type constraint).
*   **Expression Requirements:**  Specify that a certain expression involving the template parameter must be valid and well-formed at compile time, **a key feature enabling compile-time metaprogramming in AsGex.**

**Example:**

```asgex
%concept Addable<T> {
    typename Result : T; // Result type must be the same as T
    requires T + T;     // Must be able to add two values of type T
}

template <typename T : Addable<T>>
proc generic_add(a: T, b: T) -> T {
    ; Implementation will depend on the actual type T
    ; ...
}
```

**5. Threads**

**AsGex** provides language-level support for threads, enabling concurrent programming.

```ebnf
threadDefinition = "thread", identifier, [ "<", templateArgumentList, ">" ], [ "(", parameterList, ")" ], [ ":", typeReference ], "{", { topLevelElement }, "}" ;
parameterList = parameter, { ",", parameter } ;
parameter = identifier, ":", typeReference ;
```

*   Threads are defined using the `thread` keyword, followed by an identifier (the thread name), optional template arguments, optional parameters, an optional return type, and a block `{}` containing the thread's code.

**Example:**

```asgex
thread MyWorkerThread {
    ; Code to be executed in the new thread
    ; ...
}

proc main() {
    thread my_thread = MyWorkerThread ; // Create an instance of the thread
    ; ...
    threadjoin my_thread ; // Wait for the thread to finish
}
```

**6. Procedures/Functions**

Procedures (or functions) encapsulate reusable blocks of code in **AsGex**.

```ebnf
procedureDefinition = "proc", identifier, [ "(", parameterList, ")" ], [ "->", typeReference ], [ "callingconvention", identifier ], "{", { topLevelElement }, "}" ;
```

*   Procedures are defined using the `proc` keyword, followed by an identifier (the procedure name), optional parameters, an optional return type, an optional calling convention, and a block `{}` containing the procedure's code.

**Example:**

```asgex
proc multiply(a: qword, b: qword) -> qword {
    mov rax, a
    mul b
    ret
}

proc main() {
    mov r8, 5
    mov r9, 10
    call multiply
    ; rax will contain the result
}
```

**7. Instructions**

Instructions represent the fundamental operations executed by the processor in **AsGex**.

```ebnf
instruction = [ label, ":" ], [ instructionPrefix ], instructionBody, [ comment ], lineEnd ;
instructionBody = mnemonic, [ operandList ] | shorthandInstruction ;
shorthandInstruction = modifiableOperand, shorthandOperator, expression ; // RHS is a full expression
mnemonic = [ namespaceQualifier ], instructionMnemonic | templateCall ;
namespaceQualifier = identifier, "::" ;
instructionMnemonic = "mov" | "add" | "sub" | "jmp" | "call" | "ret" | "push" | "pop" | "lea" | "cmp" | "test" | "and" | "or" | "xor" | "not" | "neg" | "mul" | "imul" | "div" | "idiv" | "shl" | "shr" | "sar" | "rol" | "ror" | "rcl" | "rcr" | "jz" | "jnz" | "je" | "jne" | "js" | "jns" | "jg" | "jge" | "jl" | "jle" | "ja" | "jae" | "jb" | "jbe" ; (* Example Mnemonics *)
```

*   An instruction line can optionally start with a `label` followed by a colon.
*   It can have optional `instructionPrefixes` (e.g., `rep`, segment overrides).
*   The core of the instruction is the `instructionBody`, which can be either:
    *   A `mnemonic` (optionally namespace-qualified or a template call) followed by an optional `operandList`.
    *   A `shorthandInstruction` which simplifies common operations by combining an operator with an assignment.
*   An optional `comment` can follow the instruction body.
*   The line ends with `lineEnd` (newline or end-of-file).

**Example:**

```asgex
my_label:
    mov rax, 0x10 ; Move immediate value to register
    add rax, rbx  ; Add register rbx to rax
    jz  end_block ; Jump if zero flag is set
    ; ...
end_block:
```

**8. Directives**

Directives are instructions to the **AsGex** assembler/compiler, controlling aspects of the compilation process or data layout.

```ebnf
directive = ".", directiveName, [ directiveArguments ], [ comment ], lineEnd ;
directiveName = dataDirective | equateDirective | constDirective | incbinDirective | timesDirective | segmentDirective | useDirective | typeDirective |
                mutexDirective | conditionDirective | globalDirective | externDirective | alignDirective | sectionDirective |
                ifDirective | entryPointDirective | callingConventionDirective | acpiDirective | ioDirective | structDirective | cpuDirective | bitsDirective |
                stackDirective | warningDirective | errorDirective | includeDirective | includeOnceDirective | listDirective | nolistDirective |
                debugDirective | orgDirective | mapDirective | argDirective | localDirective | setDirective | unsetDirective |
                assertDirective | optDirective | evalDirective | repDirective | defaultDirective |
                ifdefDirective | ifndefDirective | elifdefDirective | elifndefDirective | exportDirective | commonDirective |
                fileDirective | lineDirective | contextDirective | endcontextDirective |
                allocDirective | freeDirective | bitfieldDirective | gpuDirective | uefiDirective | staticDirective | dataBlockDirective |
                gdtDirective | idtDirective | linkerDirective ;
// ... (Individual directive definitions follow - see the full grammar)
```

Directives start with a `.` followed by a `directiveName` and optional `directiveArguments`.

**More Detailed Directive Explanations:**

*   `.equ identifier, expression`:  The `equateDirective` assigns a symbolic name (`identifier`) to the result of a constant `expression`. This is a fundamental way to define constants.
*   `const identifier = constExpression`: The `constDirective` declares a compile-time constant. Similar to `.equ`, but potentially with stronger type checking and scoping.
*   `incbin stringLiteral, [offset, [length]]`: The `incbinDirective` includes the binary content of the file specified by `stringLiteral` directly into the output. Optional `offset` and `length` can be used to include only a portion of the file. Useful for embedding resources.
*   `times constExpression ( repeatableElement )`: The `timesDirective` repeats the `repeatableElement` (instruction, data definition, etc.) a specified number of times at compile time. This is a basic form of compile-time code generation.
*   `%segment identifier, [address], { topLevelElement }`: The `segmentDirective` defines a memory segment with a given `identifier` and optional starting `address`. Code and data within the block will be placed in this segment. Crucial for memory organization.
*   `use identifier [as alias]`: The `useDirective` imports symbols (constants, types, procedures) from another module or namespace. The optional `as` clause allows renaming the imported symbol.
*   `type identifier as typeDefinition`: The `typeDirective` creates a named alias for a `typeDefinition`, improving code readability and maintainability.
*   `mutex identifier`, `condition identifier`: These directives declare mutexes and condition variables for thread synchronization.
*   `global symbol, ...`: The `globalDirective` declares symbols as globally visible, making them accessible from other compilation units.
*   `extern symbol, ...`: The `externDirective` declares symbols that are defined in other compilation units.
*   `align constExpression`: The `alignDirective` ensures that the next data or code is placed at a memory address that is a multiple of `constExpression`. Important for performance.
*   `section identifier, [stringLiteral]`: The `sectionDirective` switches to a specific linker section. The optional string literal provides additional information for the linker.
*   `%if constExpression { ... } [%elif constExpression { ... }] [%else { ... }] %endif`: These directives implement conditional compilation. Code within the blocks is included or excluded based on the compile-time evaluation of the `constExpression`. **A powerful tool for compile-time metaprogramming.**
*   `entrypoint identifier`: The `entryPointDirective` specifies the starting point of program execution.
*   `callingconvention identifier`: The `callingConventionDirective` sets the default calling convention for procedures in the current scope.
*   `acpi identifier { dataDefinition }`: The `acpiDirective` likely defines data structures related to ACPI (Advanced Configuration and Power Interface), used in system programming.
*   `io (in | out) (b | w | d), operand, operand`: The `ioDirective` generates instructions for performing I/O operations.
*   `%struct identifier [{ attributeList }] { structMemberList }`: The `structDirective` defines a structure data type.
*   `cpu identifier, ...`: The `cpuDirective` specifies the target CPU architecture or features.
*   `bits (16 | 32 | 64)`: The `bitsDirective` sets the target architecture's bitness.
*   `stack constExpression`: The `stackDirective` likely reserves space for the stack.
*   `warning stringLiteral`, `error stringLiteral`: These directives cause the compiler to issue a warning or error message, respectively.
*   `include stringLiteral`, `includeonce stringLiteral`: These directives include the contents of the specified file. `includeonce` prevents multiple inclusions of the same file.
*   `list`, `nolist`: These directives control the generation of listing files during compilation.
*   `debug stringLiteral`: The `debugDirective` can be used to embed debugging information.
*   `org constExpression`: The `orgDirective` sets the origin address for subsequent code or data.
*   `map constExpression, constExpression`: The `mapDirective` might define a memory mapping.
*   `arg identifier [: typeReference]`: The `argDirective` likely declares command-line arguments for the program.
*   `local identifier [: typeReference] [= expression]`: The `localDirective` declares a local variable within a procedure.
*   `set identifier, (stringLiteral | expression)`, `unset identifier`: These directives allow setting and unsetting environment variables or compiler flags.
*   `assert constExpression [, stringLiteral]`: The `assertDirective` checks a condition at compile time. If the condition is false, a compilation error is issued.
*   `opt identifier, ...`: The `optDirective` likely enables or disables compiler optimizations.
*   `eval expression`: The `evalDirective` evaluates an expression at compile time.
*   `rep constExpression ( repeatableElement )`: Similar to `times`, but potentially used within specific contexts.
*   `default identifier = constExpression`: The `defaultDirective` might set a default value for a variable or option.
*   `ifdef identifier`, `ifndef identifier`, `elifdef identifier`, `elifndef identifier`: These are conditional compilation directives based on whether an identifier is defined.
*   `export [namespaceQualifier] identifier`: The `exportDirective` makes a symbol visible outside the current module.
*   `common identifier, constExpression`: The `commonDirective` declares a common symbol, which might be resolved by the linker.
*   `.file stringLiteral`, `.line constExpression`, `context`, `endcontext`: These directives provide information for debugging and error reporting.
*   `alloc identifier: typeReference [, constExpression]`: The `allocDirective` explicitly allocates memory.
*   `free modifiableOperand`: The `freeDirective` frees allocated memory.
*   `%bitfield identifier [: typeReference] { bitfieldMemberList }`: The `bitfieldDirective` defines a structure where members occupy a specific number of bits.
*   `gpu identifier [(directiveArgumentList)]`, `uefi identifier [(directiveArgumentList)]`: These directives likely provide specific configurations or commands for GPU and UEFI environments.
*   `static [sectionSpecifier] dataDefinition`: The `staticDirective` declares static data.
*   `data identifier { dataBlockItem }`: The `dataBlockDirective` defines a block of initialized data.
*   `gdt identifier { dataDefinition }`, `idt identifier { dataDefinition }`: These directives are used to define the Global Descriptor Table and Interrupt Descriptor Table, respectively, crucial for operating system kernels and low-level system programming.
*   `linkerDirective = "library", stringLiteral`: The `linkerDirective` provides instructions to the linker, such as specifying libraries to link against.

**9. Macros**

Macros provide a mechanism for code generation through textual substitution in **AsGex**.

```ebnf
macroDefinition = "#macro", identifier, [ "(", parameterList, ")" ], "{", { topLevelElement }, "}" ;
```

*   Macros are defined using the `#macro` keyword, followed by an identifier (the macro name), optional parameters, and a block `{}` containing the macro's body.

**Example:**

```asgex
#macro push_and_increment(reg) {
    push reg
    inc reg
}

proc main() {
    mov rcx, 10
    push_and_increment(rcx) ; Expands to 'push rcx; inc rcx;'
    ; ...
}
```

**10. Modules**

Modules provide a higher-level organization for code in **AsGex**, similar to namespaces but potentially with features for separate compilation or linking.

```ebnf
moduleDefinition = "%module", identifier, "{", { topLevelElement }, "}" ;
```

*   Modules are defined using the `%module` keyword, followed by an identifier (the module name) and a block `{}` containing the module's contents.

**Example:**

```asgex
%module MyUtils {
    const BUFFER_SIZE = 256;

    proc clear_buffer(buffer: ptr<byte>) {
        ; ...
    }
}

proc main() {
    ; Accessing the constant from the module
    local buffer: array<byte, MyUtils::BUFFER_SIZE>;
    call MyUtils::clear_buffer(buffer);
}
```

**11. Register Classes**

Register classes allow grouping related registers for easier management or abstraction in **AsGex**.

```ebnf
registerClassDefinition = "%regclass", identifier, "=", "{", registerList, "}" ;
registerList = register, { ",", register } ;
```

*   Register classes are defined using the `%regclass` keyword, an identifier, the `=` operator, and a comma-separated list of registers within curly braces.

**Example:**

```asgex
%regclass general_purpose_registers = { rax, rbx, rcx, rdx, rsi, rdi, rsp, rbp, r8, r9, r10, r11, r12, r13, r14, r15 };

proc main() {
    ; Can be used in macros or potentially for type constraints (future feature)
    ; ...
}
```

**12. Templates**

Templates enable generic programming in **AsGex** by allowing code to be written without specifying concrete types until instantiation. **This is a cornerstone of AsGex's zero-overhead abstraction philosophy.**

```ebnf
templateDefinition = "template", [ "<", templateParameterList, ">" ], identifier, [ "(", parameterList, ")" ], [ "->", typeReference ], [ requiresClause ], [ "{", attributeList, "}" ], "{", { templateElement }, "}" ;
templateParameterList = templateParameter, { ",", templateParameter } ;
templateParameter = ( "typename", identifier, [ ":", conceptReference | typeReference ] ) | ( "const", identifier, ":", typeReference, [ "=", constExpression ] ) | ( "...", identifier ) ; // Allow concept or type constraint
requiresClause = "requires", conceptConjunction ;
conceptConjunction = conceptDisjunction, { "&&", conceptDisjunction };
conceptDisjunction = conceptReference, { "||", conceptReference };
conceptReference = [ namespaceQualifier ], identifier [ "<", [templateArgumentList] ,">"] ;
templateElement = topLevelElement | unsafeBlock | staticBlock ;
unsafeBlock = "unsafe", "{", { topLevelElement }, "}" ;
staticBlock = "static", "{", { dataDefinition }, "}" ;
templateCall = [ namespaceQualifier ], identifier, "<", [ templateArgumentList ], ">" ;
templateArgumentList = templateArgument, { ",", templateArgument } ;
templateArgument = typeReference | constExpression ;
```

*   Templates are defined using the `template` keyword, followed by optional template parameters enclosed in `<>`, an identifier (the template name), optional parameters, an optional return type, an optional `requiresClause` to specify concept constraints, optional attributes, and a block `{}` containing the template's definition.

**Example:**

```asgex
template <typename T>
proc identity(value: T) -> T {
    mov rax, value ; Assuming 'mov' works generically for T
    ret
}

proc main() {
    call identity<dword>(123) ; Instantiate with dword
    call identity<qword>(0xABCDEF0123456789) ; Instantiate with qword
}
```

**13. Memory Safety Mechanisms in AsGex**

While working at the assembly level inherently requires careful memory management, **AsGex incorporates several features to enhance memory safety**:

*   **Strong Typing:** The type system, enforced at compile time, helps prevent accidental misuse of data. Explicit type declarations for variables and procedure parameters allow the compiler to catch type errors that could lead to memory corruption.
*   **`array` Type with Optional Bounds Checking:** The `array` type can optionally include `checked`, which would instruct the compiler to insert runtime checks for array access boundaries (though this might introduce some overhead).
*   **`ptr` Type with Mutability Control:** The `ptr` type allows specifying whether a pointer is `mutable` or `immutable`. Immutability can prevent unintended modifications of memory.
*   **Namespaces and Modules:** These features help organize code and reduce the chance of naming collisions, which can sometimes lead to unintended access to memory locations.
*   **Compile-Time Evaluation and Assertions:** The ability to evaluate expressions and assert conditions at compile time allows developers to catch potential memory-related errors early in the development process. For example, `assert(sizeof(my_struct) < MAX_BUFFER_SIZE)` can prevent buffer overflows.
*   **Explicit Memory Management with `alloc` and `free`:** While requiring manual memory management, the `alloc` and `free` directives provide explicit control, making memory allocation more visible and potentially easier to audit. However, this also places the burden of correctness on the programmer.
*   **`unsafe` Blocks:** The `unsafe` keyword explicitly marks sections of code where the compiler relaxes some safety checks, acknowledging the potential for memory unsafety in those areas. This encourages developers to isolate potentially unsafe operations.

**It's important to note that AsGex, being an assembly-level language, cannot guarantee complete memory safety like some higher-level languages with automatic memory management. The programmer still bears responsibility for correct memory usage, but AsGex provides tools and features to help mitigate risks.**

**14. Tooling and Ecosystem**

The planned tooling for **AsGex** includes:

*   **`asgexc` (AsGex Compiler):** The primary tool for compiling `.asgex` source files into object files or directly into executable binaries. This compiler will implement the grammar and perform the various stages of compilation, including lexical analysis, parsing, semantic analysis, optimization, and code generation.
*   **`asgexasm` (AsGex Assembler):**  Potentially a separate assembler for assembling `.s` files generated by `asgexc` or written manually.
*   **`asgexlink` (AsGex Linker):** A linker for combining object files and libraries into executable programs.
*   **Debugger Integration:**  Integration with common debugging tools will be a priority.

The ecosystem is envisioned to include standard libraries for common tasks, potentially leveraging existing C libraries through a foreign function interface (FFI).

**15. Illustrative Metaprogramming Examples**

**Example 1: Compile-Time Loop Unrolling**

```asgex
template <const N: dword>
proc unrolled_add(ptr: ptr<dword>) {
    times N {
        add dword ptr [ptr], 1
        add ptr, 4
    }
}

proc main() {
    local my_array: array<dword, 4>;
    call unrolled_add<4>(my_array); ; The loop will be fully unrolled at compile time
}
```

**Example 2: Conditional Code Generation based on Architecture**

```asgex
%if __TARGET_ARCH__ == "x64" {
    ; Generate x64-specific instructions
    mov rax, 10
} %elif __TARGET_ARCH__ == "x86" {
    ; Generate x86-specific instructions
    mov eax, 10
} %endif
```

**Example 3: Generating Data Structures Based on Compile-Time Constants**

```asgex
template <const SIZE: dword>
%struct StaticBuffer {
    data: array<byte, SIZE>
}

const SMALL_BUFFER_SIZE = 64;
const LARGE_BUFFER_SIZE = 1024;

type SmallBufferType = StaticBuffer<SMALL_BUFFER_SIZE>;
type LargeBufferType = StaticBuffer<LARGE_BUFFER_SIZE>;
```

These examples showcase how **AsGex's** compile-time features can be used for optimization, platform-specific code generation, and creating flexible data structures.

**16. Conclusion**

The **AsGex** Assembly Language provides a rich and powerful syntax for low-level programming with modern features. **Its emphasis on zero-overhead abstractions and compile-time metaprogramming allows developers to write efficient and highly customized code.** Its comprehensive set of directives and instruction mnemonics allows for precise control over hardware while promoting code organization and maintainability. The integrated memory safety features, while not absolute, help mitigate common errors. **AsGex** is well-suited for performance-critical and safety-sensitive applications across various target architectures, empowering developers to push the boundaries of low-level programming.
