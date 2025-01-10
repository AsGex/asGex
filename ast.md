
```markdown
# AsGex Binary Abstract Syntax Tree (AST) Representation

This document details the binary Abstract Syntax Tree (AST) representation for the AsGex assembly language. This AST serves as a structured, machine-readable representation of AsGex source code, facilitating subsequent compiler phases such as semantic analysis, optimization, and code generation. The design prioritizes a faithful mapping to the grammar, enabling a comprehensive and unambiguous representation while maintaining efficiency and compactness.

## Core Node Structure

The fundamental building block of the AST is the `AstNode` structure:

```c++
struct AstNode {
    uint16_t type;      // Type identifier (see NodeType enumeration)
    uint8_t flags;     // Bit flags for boolean attributes
    uint32_t data_offset; // Offset into data buffer for additional information
};
```

*   **`type`:** A `uint16_t` value from the `NodeType` enumeration, uniquely identifying the grammatical element this node represents (e.g., `NT_INSTRUCTION`, `NT_IDENTIFIER`, `NT_DIRECTIVE`).
*   **`flags`:** An `uint8_t` field used as a bitmask to store boolean properties relevant to the node. These flags encode information crucial for compiler optimizations and analysis.
*   **`data_offset`:** A `uint32_t` offset into a separate, contiguous data buffer. This offset indicates the starting location of data associated with the node (e.g., string literals, numerical values, or sequences of child `AstNode` structures). A value of 0 signifies no associated data beyond the inherent structure.

## NodeType Enumeration

The `NodeType` enumeration provides a distinct identifier for each significant non-terminal symbol in the AsGex grammar. This ensures a one-to-one mapping between grammar constructs and AST node types.

```c++
enum NodeType : uint16_t {
    // Program Structure
    NT_PROGRAM,
    NT_TOP_LEVEL_ELEMENT,

    // Namespaces
    NT_NAMESPACE_DEFINITION,

    // Concepts
    NT_CONCEPT_DEFINITION,
    NT_CONCEPT_REQUIREMENT,
    NT_TYPE_REQUIREMENT,
    NT_EXPRESSION_REQUIREMENT,
    NT_WHERE_CLAUSE,

    // Threads
    NT_THREAD_DEFINITION,
    NT_PARAMETER_LIST,
    NT_PARAMETER,

    // Procedures/Functions
    NT_PROCEDURE_DEFINITION,

    // Instructions
    NT_INSTRUCTION,
    NT_INSTRUCTION_BODY,
    NT_SHORTHAND_INSTRUCTION,
    NT_MNEMONIC,
    NT_NAMESPACE_QUALIFIER,
    NT_INSTRUCTION_PREFIX,
    NT_REPEAT_PREFIX_KIND,
    NT_SEGMENT_PREFIX_KIND,
    NT_ADDRESS_PREFIX_KIND,
    NT_DATA_PREFIX_KIND,
    NT_VECTOR_PREFIX_KIND,
    NT_OTHER_PREFIX_KIND,
    NT_INSTRUCTION_PREFIX_GROUP,

    // Mnemonics
    NT_MNEMONIC_MOV,
    NT_MNEMONIC_ADD,
    NT_MNEMONIC_SUB,
    NT_MNEMONIC_JMP,
    NT_MNEMONIC_CALL,
    NT_MNEMONIC_RET,
    NT_MNEMONIC_PUSH,
    NT_MNEMONIC_POP,
    NT_MNEMONIC_LEA,
    NT_MNEMONIC_CMP,
    NT_MNEMONIC_TEST,
    NT_MNEMONIC_AND,
    NT_MNEMONIC_OR,
    NT_MNEMONIC_XOR,
    NT_MNEMONIC_NOT,
    NT_MNEMONIC_NEG,
    NT_MNEMONIC_MUL,
    NT_MNEMONIC_IMUL,
    NT_MNEMONIC_DIV,
    NT_MNEMONIC_IDIV,
    NT_MNEMONIC_SHL,
    NT_MNEMONIC_SHR,
    NT_MNEMONIC_SAR,
    NT_MNEMONIC_ROL,
    NT_MNEMONIC_ROR,
    NT_MNEMONIC_RCL,
    NT_MNEMONIC_RCR,
    NT_MNEMONIC_JZ,
    NT_MNEMONIC_JNZ,
    NT_MNEMONIC_JE,
    NT_MNEMONIC_JNE,
    NT_MNEMONIC_JS,
    NT_MNEMONIC_JNS,
    NT_MNEMONIC_JG,
    NT_MNEMONIC_JGE,
    NT_MNEMONIC_JL,
    NT_MNEMONIC_JLE,
    NT_MNEMONIC_JA,
    NT_MNEMONIC_JAE,
    NT_MNEMONIC_JB,
    NT_MNEMONIC_JBE,

    // Directives
    NT_DIRECTIVE,
    NT_DIRECTIVE_NAME,
    NT_DIRECTIVE_ARGUMENTS,

    // Directive Names
    NT_DIRECTIVE_DATA,
    NT_DIRECTIVE_EQUATE,
    NT_DIRECTIVE_CONST,
    NT_DIRECTIVE_INCBIN,
    NT_DIRECTIVE_TIMES,
    NT_DIRECTIVE_SEGMENT,
    NT_DIRECTIVE_USE,
    NT_DIRECTIVE_TYPE,
    NT_DIRECTIVE_MUTEX,
    NT_DIRECTIVE_CONDITION,
    NT_DIRECTIVE_GLOBAL,
    NT_DIRECTIVE_EXTERN,
    NT_DIRECTIVE_ALIGN,
    NT_DIRECTIVE_SECTION,
    NT_DIRECTIVE_IF,
    NT_DIRECTIVE_ELIF,
    NT_DIRECTIVE_ELSE,
    NT_DIRECTIVE_ENDIF,
    NT_DIRECTIVE_IFDEF,
    NT_DIRECTIVE_IFNDEF,
    NT_DIRECTIVE_ELIFDEF,
    NT_DIRECTIVE_ELIFNDEF,
    NT_DIRECTIVE_ENTRYPOINT,
    NT_DIRECTIVE_CALLINGCONVENTION,
    NT_DIRECTIVE_ACPI,
    NT_DIRECTIVE_IO,
    NT_DIRECTIVE_STRUCT_DIRECTIVE, // Renamed to avoid ambiguity with struct definition
    NT_DIRECTIVE_CPU,
    NT_DIRECTIVE_BITS,
    NT_DIRECTIVE_STACK,
    NT_DIRECTIVE_WARNING,
    NT_DIRECTIVE_ERROR,
    NT_DIRECTIVE_INCLUDE,
    NT_DIRECTIVE_INCLUDEONCE,
    NT_DIRECTIVE_LIST,
    NT_DIRECTIVE_NOLIST,
    NT_DIRECTIVE_DEBUG,
    NT_DIRECTIVE_ORG,
    NT_DIRECTIVE_MAP,
    NT_DIRECTIVE_ARG,
    NT_DIRECTIVE_LOCAL,
    NT_DIRECTIVE_SET,
    NT_DIRECTIVE_UNSET,
    NT_DIRECTIVE_ASSERT,
    NT_DIRECTIVE_OPT,
    NT_DIRECTIVE_EVAL,
    NT_DIRECTIVE_REP,
    NT_DIRECTIVE_DEFAULT,
    NT_DIRECTIVE_EXPORT,
    NT_DIRECTIVE_COMMON,
    NT_DIRECTIVE_FILE,
    NT_DIRECTIVE_LINE,
    NT_DIRECTIVE_CONTEXT,
    NT_DIRECTIVE_ENDCONTEXT,
    NT_DIRECTIVE_ALLOC,
    NT_DIRECTIVE_FREE,
    NT_DIRECTIVE_BITFIELD_DIRECTIVE, // Renamed for clarity
    NT_DIRECTIVE_GPU,
    NT_DIRECTIVE_UEFI,
    NT_DIRECTIVE_STATIC,
    NT_DIRECTIVE_DATABLOCK,
    NT_DIRECTIVE_GDT,
    NT_DIRECTIVE_IDT,
    NT_DIRECTIVE_LINKER,

    // ARM Directives
    NT_ARM_DIRECTIVE,
    NT_ARM_DIRECTIVE_NAME,
    NT_ARM_DIRECTIVE_ARGUMENTS,
    NT_ARM_DIRECTIVE_SYNTAX,
    NT_ARM_DIRECTIVE_ARCH,
    NT_ARM_DIRECTIVE_THUMB,
    NT_ARM_DIRECTIVE_ARM,
    NT_ARM_DIRECTIVE_GLOBAL,
    NT_ARM_DIRECTIVE_FUNC,
    NT_ARM_DIRECTIVE_ENDFUNC,
    NT_ARM_DIRECTIVE_TYPE,
    NT_ARM_DIRECTIVE_SIZE,

    // Linker Directives
    NT_LINKER_DIRECTIVE,

    // GDT/IDT Directives
    NT_GDT_DIRECTIVE,
    NT_IDT_DIRECTIVE,

    // GPU/UEFI Specific Directives
    NT_GPU_DIRECTIVE,
    NT_UEFI_DIRECTIVE,

    // Static Directive
    NT_STATIC_DIRECTIVE,
    NT_SECTION_SPECIFIER,
    NT_DATA_DEFINITION,

    // Data Block Directive
    NT_DATA_BLOCK_DIRECTIVE,
    NT_DATA_BLOCK_ITEM,

    // Common Directive Components
    NT_DIRECTIVE_ARGUMENT,
    NT_DIRECTIVE_ARGUMENT_LIST,

    // Data Directives
    NT_DATA_DIRECTIVE_KIND, // Represents "db", "dw", etc.
    NT_DATA_LIST,
    NT_DATA_VALUE,

    // Type Directives
    NT_TYPE_DIRECTIVE,
    NT_TYPE_DEFINITION,
    NT_BASIC_TYPE,
    NT_ARRAY_TYPE,
    NT_POINTER_TYPE,
    NT_TYPE_REFERENCE,
    NT_STRUCT_REFERENCE,
    NT_ENUM_REFERENCE,
    NT_TEMPLATE_REFERENCE,
    NT_TYPE_REFERENCE_LIST,

    // Enum Definition
    NT_ENUM_DEFINITION,
    NT_ENUM_MEMBER_LIST,
    NT_ENUM_MEMBER,

    // Struct Definition
    NT_STRUCT_DEFINITION,
    NT_STRUCT_MEMBER_LIST,
    NT_STRUCT_MEMBER,

    // Bitfield Directive & Definition
    NT_BITFIELD_DIRECTIVE,
    NT_BITFIELD_MEMBER_LIST,
    NT_BITFIELD_MEMBER,

    // Attributes
    NT_ATTRIBUTE_LIST,
    NT_ATTRIBUTE,
    NT_CONST_EXPRESSION_LIST,

    // ARM Macros
    NT_ARM_MACRO_DEFINITION,
    NT_ARM_PARAMETER_LIST,
    NT_ARM_PARAMETER,

    // Macros
    NT_MACRO_DEFINITION,

    // Modules
    NT_MODULE_DEFINITION,

    // Register Classes
    NT_REGISTER_CLASS_DEFINITION,
    NT_REGISTER_LIST,

    // Templates
    NT_TEMPLATE_DEFINITION,
    NT_TEMPLATE_PARAMETER_LIST,
    NT_TEMPLATE_PARAMETER,
    NT_REQUIRES_CLAUSE,
    NT_CONCEPT_CONJUNCTION,
    NT_CONCEPT_DISJUNCTION,
    NT_CONCEPT_REFERENCE,
    NT_TEMPLATE_ELEMENT,
    NT_UNSAFE_BLOCK,
    NT_STATIC_BLOCK,
    NT_TEMPLATE_CALL,
    NT_TEMPLATE_ARGUMENT_LIST,
    NT_TEMPLATE_ARGUMENT,

    // Comments
    NT_COMMENT,

    // Labels
    NT_LABEL,

    // ARM Instruction Prefixes
    NT_ARM_INSTRUCTION_PREFIX,
    NT_ARM_CONDITION_CODE,
    NT_ARM_HINT_PREFIX,

    // Shorthand Operations
    NT_SHORTHAND_OPERATOR_KIND,

    // Thread Operations
    NT_THREAD_CREATION,
    NT_EXPRESSION_LIST,
    NT_THREAD_DIRECTIVE,
    NT_THREAD_DIRECTIVE_KIND,
    NT_THREAD_JOIN_DIRECTIVE,
    NT_THREAD_TERMINATE_DIRECTIVE,
    NT_THREAD_SLEEP_DIRECTIVE,
    NT_THREAD_YIELD_DIRECTIVE,
    NT_THREAD_LOCAL_DIRECTIVE,

    // Operands
    NT_OPERAND_LIST,
    NT_OPERAND,
    NT_OPERAND_SIZE_OVERRIDE,
    NT_OPERAND_TYPE,
    NT_OPERAND_KIND,
    NT_MODIFIABLE_OPERAND,

    // ARM Operands
    NT_ARM_OPERAND_LIST,
    NT_ARM_OPERAND,
    NT_ARM_OPERAND_SIZE_OVERRIDE,
    NT_ARM_OPERAND_KIND,
    NT_ARM_MODIFIABLE_OPERAND,
    NT_ARM_MEMORY_OPERAND,
    NT_ARM_ADDRESS,
    NT_ARM_ADDRESS_BASE,
    NT_ARM_ADDRESS_OFFSET,
    NT_ARM_ADDRESS_DISPLACEMENT,
    NT_ARM_ADDRESS_SCALE_INDEX,
    NT_ARM_SHIFTED_REGISTER,
    NT_SHIFT_TYPE,
    NT_SHIFT_OPERATION,
    NT_SYMBOL_REFERENCE,
    NT_ADDRESS_TERM,

    // GPU Operands
    NT_GPU_OPERAND_LIST,
    NT_GPU_OPERAND,
    NT_GPU_OPERAND_SIZE_OVERRIDE,
    NT_GPU_OPERAND_KIND,
    NT_GPU_MODIFIABLE_OPERAND,
    NT_GPU_MEMORY_OPERAND,
    NT_GPU_ADDRESS,
    NT_GPU_ADDRESS_SPACE,
    NT_GPU_ADDRESS_EXPRESSION,

    // Operand Kinds
    NT_IMMEDIATE,
    NT_REGISTER_OPERAND,
    NT_MEMORY_OPERAND,

    // Registers
    NT_REGISTER,
    NT_GENERAL_REGISTER_KIND,
    NT_SEGMENT_REGISTER_KIND,
    NT_CONTROL_REGISTER_KIND,
    NT_DEBUG_REGISTER_KIND,
    NT_MMX_REGISTER_KIND,
    NT_XMM_REGISTER_KIND,
    NT_YMM_REGISTER_KIND,
    NT_ZMM_REGISTER_KIND,
    NT_ARM_REGISTER_KIND,
    NT_GPU_REGISTER_KIND,

    // Constants
    NT_CONSTANT,
    NT_NUMBER,
    NT_HEX_NUMBER,
    NT_BIN_NUMBER,
    NT_FLOAT_NUMBER,
    NT_CHARACTER,
    NT_ESCAPE_SEQUENCE,
    NT_ADDRESS_LITERAL,

    // Expressions
    NT_EXPRESSION,
    NT_CONDITIONAL_EXPRESSION,
    NT_LOGICAL_OR_EXPRESSION,
    NT_LOGICAL_AND_EXPRESSION,
    NT_BITWISE_OR_EXPRESSION,
    NT_BITWISE_XOR_EXPRESSION,
    NT_BITWISE_AND_EXPRESSION,
    NT_SHIFT_EXPRESSION,
    NT_ADDITIVE_EXPRESSION,
    NT_MULTIPLICATIVE_EXPRESSION,
    NT_UNARY_EXPRESSION,
    NT_TYPE_CONVERSION,
    NT_SIZEOF_EXPRESSION,
    NT_ALIGNOF_EXPRESSION,

    // Memory Addresses
    NT_MEMORY_ADDRESS,
    NT_ADDRESS_BASE,
    NT_ADDRESS_OFFSET,
    NT_ADDRESS_DISPLACEMENT,
    NT_ADDRESS_SCALE_INDEX,
    NT_ADDRESS_TERM,
    NT_SCALE_FACTOR,

    // String Literals
    NT_STRING_LITERAL,

    // Lexical Tokens
    NT_IDENTIFIER,
    NT_DIGIT,
    NT_HEX_DIGIT,
    NT_BIN_DIGIT,
    NT_EOF
};
```

## Flags Usage Details

The `flags` byte within the `AstNode` structure encodes specific boolean properties. The interpretation of these bits depends on the `NodeType`.

| NodeType                          | Bit 0 (0x01)           | Bit 1 (0x02)           | Bit 2 (0x04)           | Bit 3 (0x08)           | Bit 4 (0x10)               | Bit 5 (0x20)               | Bit 6 (0x40)                 | Bit 7 (0x80)                     |
| :-------------------------------- | :--------------------- | :--------------------- | :--------------------- | :--------------------- | :------------------------- | :------------------------- | :--------------------------- | :------------------------------- |
| `NT_INSTRUCTION`                  | `has_label`            | `has_prefixes`         | `has_operands`         |                        | `has_implicit_side_effects` | `can_be_eliminated`        | `is_conditional_jump`      |                                  |
| `NT_IDENTIFIER`                   | `is_namespace_qualified` | `is_macro_parameter`   |                        |                        |                            |                            |                            |                                  |
| `NT_CONSTANT`                     | `is_negative`          | `is_hex`               | `is_binary`            | `is_float`             | `is_character`             | `is_string_literal`        | `can_be_folded`              |                                  |
| `NT_DIRECTIVE`                    | `is_global`            | `is_extern`              | `is_thread_local`        |                        |                            |                            |                            |                                  |
| `NT_REPEAT_PREFIX_KIND`           | `is_repe`              | `is_repz`              | `is_repne`             | `is_repnz`             | `is_lock`                  |                            |                            |                                  |
| `NT_SHORTHAND_OPERATOR_KIND`      | `is_assign`            | `is_plus_eq`           | `is_minus_eq`          | `is_mul_eq`            | `is_div_eq`                | `is_and_eq`                | `is_or_eq`                   | `is_xor_eq`                      |
| `NT_THREAD_CREATION`              |                        |                        |                        |                        |                            |                            | `has_side_effects`           |                                  |
| `NT_MEMORY_OPERAND`               | `is_read`              | `is_write`               |                        |                        |                            |                            |                            |                                  |
| `NT_CONDITIONAL_EXPRESSION`       |                        |                        |                        |                        | `has_side_effects_true`    | `has_side_effects_false`   | `is_compile_time_constant` |                                  |
| `NT_UNARY_EXPRESSION`             |                        |                        |                        |                        |                            |                            | `is_compile_time_constant` |                                  |
| `NT_EXPRESSION` (General)       |                        |                        |                        |                        | `has_side_effects`         |                            | `is_compile_time_constant` |                                  |
| `NT_REGISTER`                     | `is_x86`                |  `is_arm`              | `is_gpu`                |                        |                            |                            |                            |                                  |
| `NT_ARM_REGISTER_KIND`            | `is_general`           | `is_sp`                | `is_lr`                | `is_pc`                | `is_apsr`                  | `is_cpsr`                  | `is_spsr`                    |                                  |
| `NT_GPU_REGISTER_KIND`           | `is_general`           | `is_float`             | `is_predicate`         | `is_thread_id`         | `is_block_id`              | `is_num_threads`           |                            |                                  |
| `NT_OPERAND`                     |                        |                        |                        |                        | `is_x86`                   | `is_arm`                   | `is_gpu`                     |                                  |
| `NT_ARM_INSTRUCTION`              | `has_label`            | `has_prefixes`         | `has_operands`         |                        | `has_implicit_side_effects` | `can_be_eliminated`        | `is_conditional_branch`      |                                  |
| `NT_GPU_INSTRUCTION`              | `has_label`            | `has_prefixes`         | `has_operands`         |                        | `has_implicit_side_effects` | `can_be_eliminated`        | `is_conditional_branch`      |                                  |
| `NT_ARM_CONDITION_CODE`          | `is_eq`                 | `is_ne`                 | `is_cs`                 | `is_cc`                 | `is_mi`                     | `is_pl`                     | `is_vs`                      |  `is_vc`                        |
| `NT_ARM_HINT_PREFIX`              | `is_wfi`                 | `is_sev`                 | `is_yield`                 | `is_nop`                 |                            |                            |                            |                                  |
| `NT_ARM_CONDITION_CODE` (Cont.)   | `is_hi`                 | `is_ls`                 | `is_ge`                 | `is_lt`                 | `is_gt`                     | `is_le`                     | `is_al`                      |                                  |

**Reserved Bits:**

*   In general, bits not assigned in the table above are **reserved for future use**.

## Data Storage Details

The `data_offset` field provides a mechanism to associate additional data with an `AstNode`. The type and interpretation of this data are specific to the `NodeType`.

| NodeType                          | Data at `data_offset`                                                                                                                                                                                                                                            |
| :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `NT_IDENTIFIER`                   | Offset to a null-terminated string in the string interning pool representing the identifier's name.                                                                                                                                                            |
| `NT_STRING_LITERAL`               | Offset to a null-terminated string in the string interning pool representing the literal value.                                                                                                                                                                |
| `NT_NUMBER`                       | Little-endian representation of the numerical value. The size is context-dependent (e.g., determined by size specifiers or default sizes).                                                                                                                       |
| `NT_HEX_NUMBER`, `NT_BIN_NUMBER`  | Little-endian representation of the numerical value.                                                                                                                                                                                                                |
| `NT_FLOAT_NUMBER`                 | Little-endian representation of the floating-point value (typically IEEE 754 single or double precision).                                                                                                                                                         |
| `NT_CHARACTER`                    | The ASCII value of the character.                                                                                                                                                                                                                            |
| `NT_MNEMONIC`                     | Offset to an `AstNode` of type `NT_MNEMONIC_*` representing the specific mnemonic.                                                                                                                                                                             |
| `NT_DIRECTIVE_NAME`               | Offset to an `AstNode` of type `NT_DIRECTIVE_*` representing the specific directive.                                                                                                                                                                           |
| `NT_INSTRUCTION`                  | Offset to a sequence of `AstNode` representing instruction prefixes (if any) and the instruction body. The sequence is terminated by an `AstNode` with `type` set to `NT_EOF`.                                                                                      |
| `NT_INSTRUCTION_BODY`             | Offset to an `AstNode` representing either a `NT_MNEMONIC` followed by an optional `NT_OPERAND_LIST`, or a `NT_SHORTHAND_INSTRUCTION`.                                                                                                                             |
| `NT_OPERAND_LIST`                 | Offset to a contiguous sequence of `AstNode` structures, each representing an individual operand. The sequence is terminated by an `AstNode` with `type` set to `NT_EOF`.                                                                                            |
| `NT_DIRECTIVE`                    | Offset to a sequence of `AstNode` representing the directive name and its arguments. The sequence is terminated by an `AstNode` with `type` set to `NT_EOF`.                                                                                                        |
| `NT_DIRECTIVE_ARGUMENTS`          | Offset to a contiguous sequence of `AstNode` structures, each representing a directive argument. The sequence is terminated by an `AstNode` with `type` set to `NT_EOF`.                                                                                           |
| `NT_PARAMETER_LIST`               | Offset to a contiguous sequence of `AstNode` structures, each representing a parameter. The sequence is terminated by an `AstNode` with `type` set to `NT_EOF`.                                                                                                     |
| `NT_TEMPLATE_PARAMETER_LIST`      | Offset to a contiguous sequence of `AstNode` structures, each representing a template parameter. The sequence is terminated by an `AstNode` with `type` set to `NT_EOF`.                                                                                              |
| `NT_LABEL`                        | Offset to a null-terminated string in the string interning pool representing the label's name.                                                                                                                                                                 |
| `NT_EXPRESSION`                    | Offset to the root of the expression subtree. The structure of the subtree depends on the specific expression type (e.g., `NT_BINARY_EXPRESSION`, `NT_UNARY_EXPRESSION`).                                                                                          |
| `NT_MEMORY_OPERAND`               | Offset to an `AstNode` of type `NT_MEMORY_ADDRESS`, representing the memory address.                                                                                                                                                                       |
| `NT_REGISTER_OPERAND`             | Offset to an `AstNode` of type `NT_REGISTER`, representing the register.                                                                                                                                                                                      |
| `NT_IMMEDIATE`                     | Offset to an `AstNode` of type `NT_CONSTANT`, representing the immediate value.                                                                                                                                                                                |
| `NT_MEMORY_ADDRESS`               | Offset to a sequence of `AstNode` representing the segment prefix (optional), address base, address offset (optional). The sequence is terminated by an `AstNode` with `type` set to `NT_EOF`.                                                                    |
| `NT_ADDRESS_BASE`                 | Offset to an `AstNode` representing either a `NT_REGISTER_OPERAND`, `NT_SYMBOL_REFERENCE`, or a sequence for relative addressing (`NT_OPERAND_KIND` followed by `NT_SYMBOL_REFERENCE`).                                                                            |
| `NT_ADDRESS_OFFSET`               | Offset to an `AstNode` representing either a `NT_ADDRESS_DISPLACEMENT` or `NT_ADDRESS_SCALE_INDEX`.                                                                                                                                                          |
| `NT_ARM_ADDRESS`               | Offset to a sequence of `AstNode` representing the base register and optional offset components. Terminated by an `AstNode` with `type` set to `NT_EOF`.                                                                                            |
| `NT_ARM_ADDRESS_BASE`                | Offset to an `AstNode` representing either a `NT_ARM_REGISTER_KIND`, `NT_SYMBOL_REFERENCE`, or a sequence for relative addressing. (`NT_OPERAND_KIND` followed by `NT_SYMBOL_REFERENCE`). |
| `NT_ARM_ADDRESS_OFFSET`               | Offset to an `AstNode` representing either a `NT_ARM_ADDRESS_DISPLACEMENT`, `NT_ARM_ADDRESS_SCALE_INDEX` or a `NT_ARM_SHIFTED_REGISTER`.                                                                                                                                                          |
| `NT_GPU_ADDRESS`               | Offset to a sequence of `AstNode` representing the address space (optional) and address expression. Terminated by an `AstNode` with `type` set to `NT_EOF`.                                                                                            |
| `NT_GPU_ADDRESS_EXPRESSION`                | Offset to an `AstNode` representing either a `NT_GPU_REGISTER_KIND`, a combination of a register and immediate/register, a `NT_SYMBOL_REFERENCE` or a `NT_EXPRESSION`. |
| `NT_GPU_OPERAND`                | Offset to an `AstNode` representing `NT_GPU_OPERAND_SIZE_OVERRIDE` (optional), followed by `NT_GPU_OPERAND_KIND`. |
| `NT_ARM_OPERAND`                | Offset to an `AstNode` representing `NT_ARM_OPERAND_SIZE_OVERRIDE` (optional), followed by `NT_ARM_OPERAND_KIND`. |

## String Interning

All strings (identifiers, string literals, labels) are stored in a dedicated **string interning pool**. This pool is a contiguous block of memory where each string is stored as a **null-terminated** sequence of characters. The `data_offset` for nodes that refer to strings will point to the start of the corresponding string within this pool.

**Benefits of String Interning:**

*   **Reduced Memory Usage:** Duplicate strings are stored only once, saving memory.
*   **Faster String Comparisons:** String comparisons can be performed by comparing pointers instead of character-by-character comparison.

## Memory Management

*   **Data Buffer Allocation:** The data buffer is allocated as a single contiguous block of memory during the AST construction phase. Its size is determined dynamically based on the size of the input program.
*   **AST Node Allocation:** `AstNode` structures are also allocated from a contiguous block of memory.
*   **Ownership and Lifetimes:** The parser/compiler is responsible for managing the memory used by the AST and the data buffer. The memory is typically freed after the AST is no longer needed (e.g., after code generation).

**Endianness:**

All multi-byte values in the AST representation (`type`, `flags`, `data_offset`, and data in the buffer) are stored in **little-endian** byte order.

**Versioning:**

The current AST representation has version number **1**. The version number is stored in a separate header preceding the AST data. This allows for future modifications to the AST format while maintaining backward compatibility.

## Error Handling

*   **Invalid AST Nodes:** During parsing, if an error is encountered, a special `NodeType` value, `NT_ERROR`, is used to create an error node. The `data_offset` of this error node can point to an error message string in the string interning pool or a more complex error structure.
*   **Error Recovery:** The parser may attempt basic error recovery to continue parsing and identify further errors. The specific error recovery strategy is implementation-dependent.

## Example

```assembly
.equ counter, 10
start:
    mov eax, counter
```

**Conceptual Binary AST Representation:**

```
// String Interning Pool (Example)
Offset 0x1000: "counter\0"
Offset 0x1008: "start\0"
Offset 0x1010: ".equ\0"
Offset 0x1016: "mov\0"
Offset 0x101a: "eax\0"

// Binary AST Data Buffer
Offset 0x2000: // Version Header
    0x01 0x00 0x00 0x00 // Version 1

Offset 0x2004: // NT_PROGRAM
    AstNode { type: NT_PROGRAM, flags: 0x00, data_offset: 0x200C }

Offset 0x200C: // Sequence of NT_TOP_LEVEL_ELEMENT
    // .equ counter, 10
    AstNode { type: NT_TOP_LEVEL_ELEMENT, flags: 0x00, data_offset: 0x2014 }
    // start: mov eax, counter
    AstNode { type: NT_TOP_LEVEL_ELEMENT, flags: 0x00, data_offset: 0x2034 }
    // End of sequence
    AstNode { type: NT_EOF, flags: 0x00, data_offset: 0x0000 }

Offset 0x2014: // NT_DIRECTIVE
    AstNode { type: NT_DIRECTIVE, flags: 0x00, data_offset: 0x201C }

Offset 0x201C: // Details of .equ counter, 10
    AstNode { type: NT_DIRECTIVE_EQUATE, flags: 0x00, data_offset: 0x1010 } // Pointer to ".equ"
    AstNode { type: NT_IDENTIFIER, flags: 0x00, data_offset: 0x1000 }    // Pointer to "counter"
    AstNode { type: NT_CONSTANT, flags: 0x40, data_offset: 0x202C }      // Constant 10, can_be_folded
    AstNode { type: NT_EOF, flags: 0x00, data_offset: 0x0000 }

Offset 0x202C: // Raw data for constant 10
    0x0A 0x00 0x00 0x00 // Little-endian representation

Offset 0x2034: // NT_INSTRUCTION
    AstNode { type: NT_INSTRUCTION, flags: 0x01, data_offset: 0x203C } // has_label flag set

Offset 0x203C: // Details of the instruction
    AstNode { type: NT_LABEL, flags: 0x00, data_offset: 0x1008 }       // Pointer to "start"
    AstNode { type: NT_INSTRUCTION_BODY, flags: 0x00, data_offset: 0x2044 }
    AstNode { type: NT_EOF, flags: 0x00, data_offset: 0x0000 }

Offset 0x2044: // Instruction body
    AstNode { type: NT_MNEMONIC, flags: 0x00, data_offset: 0x204C }
    AstNode { type: NT_OPERAND_LIST, flags: 0x00, data_offset: 0x2050 }

Offset 0x204C:
    AstNode { type: NT_MNEMONIC_MOV, flags: 0x00, data_offset: 0x1016 }  // Pointer to "mov"

Offset 0x2050: // Operand list
    AstNode { type: NT_OPERAND, flags: 0x00, data_offset: 0x2058 }
    AstNode { type: NT_OPERAND, flags: 0x00, data_offset: 0x2060 }
    AstNode { type: NT_EOF, flags: 0x00, data_offset: 0x0000 }

Offset 0x2058: // Operand "eax"
    AstNode { type: NT_REGISTER_OPERAND, flags: 0x01, data_offset: 0x2068 } // is_x86

Offset 0x2060: // Operand "counter"
    AstNode { type: NT_IDENTIFIER, flags: 0x00, data_offset: 0x1000 }    // Pointer to "counter"

Offset 0x2068: // NT_REGISTER
    AstNode { type: NT_REGISTER, flags: 0x01, data_offset: 0x2070 } // is_x86

Offset 0x2070: // NT_GENERAL_REGISTER_KIND
    AstNode { type: NT_GENERAL_REGISTER_KIND, flags: 0x00, data_offset: 0x101a } // Pointer to "eax"

// ... (and so on)
```

... (Continuing from the previous section)

## Conclusion

This document has outlined the binary AST representation for the AsGex assembly language. This representation provides a compact, efficient, and well-defined format for storing and manipulating AsGex programs within a compiler. The use of a `NodeType` enumeration, bit flags, and a separate data buffer with string interning allows for a faithful and optimized representation of the source code's structure and semantics. This well-structured AST is crucial for enabling subsequent compiler phases, including:

*   **Semantic Analysis:** The AST can be traversed to perform type checking, scope resolution, and other semantic checks to ensure the program's correctness.
*   **Optimization:** The flags within the `AstNode` structure, along with the structural information in the AST, provide valuable information for various compiler optimizations, such as:
    *   **Constant Folding:** Expressions involving only constants can be evaluated at compile time.
    *   **Dead Code Elimination:** Instructions or code blocks that have no effect on the program's output can be removed.
    *   **Instruction Scheduling:** Instructions can be reordered to improve performance on the target architecture.
    *   **Register Allocation:** The AST can be analyzed to determine the most efficient way to allocate registers to variables and intermediate values.
*   **Code Generation:** The AST provides a clear and unambiguous representation of the program, which can be used to generate machine code for the target architecture.

## Future Enhancements

*   **More Sophisticated Error Representation:** The current error handling mechanism can be enhanced by providing more detailed error information, such as the specific location of the error in the source code (line and column numbers) and the type of error encountered. This can be achieved by adding more fields to the `NT_ERROR` node or by creating a separate error list data structure.
*   **Attribute Grammars:** Consider integrating attribute grammars to further enhance the AST. Attribute grammars allow associating attributes (data) with grammar symbols and defining rules for computing these attributes. This can be used to perform complex semantic analysis and code transformations directly on the AST.
*   **Incremental Parsing:** For large AsGex programs, consider implementing incremental parsing, where only the modified parts of the source code are re-parsed and the AST is updated accordingly. This can significantly improve compilation speed for large projects.
*   **AST Serialization:** Implement a mechanism to serialize and deserialize the AST to and from disk. This would enable saving and restoring the AST, which can be useful for debugging, analysis, or code refactoring tools.

## Appendix: Complete Grammar Mapping to NodeType

The following table provides a complete mapping between the non-terminal symbols in the AsGex grammar and the corresponding `NodeType` values in the AST.

| Grammar Non-terminal               | NodeType                      |
| :--------------------------------- | :--------------------------- |
| `program`                          | `NT_PROGRAM`                  |
| `topLevelElement`                  | `NT_TOP_LEVEL_ELEMENT`          |
| `namespaceDefinition`              | `NT_NAMESPACE_DEFINITION`      |
| `conceptDefinition`                | `NT_CONCEPT_DEFINITION`        |
| `conceptRequirement`               | `NT_CONCEPT_REQUIREMENT`       |
| `typeRequirement`                  | `NT_TYPE_REQUIREMENT`          |
| `expressionRequirement`            | `NT_EXPRESSION_REQUIREMENT`    |
| `whereClause`                      | `NT_WHERE_CLAUSE`              |
| `threadDefinition`                 | `NT_THREAD_DEFINITION`         |
| `parameterList`                    | `NT_PARAMETER_LIST`            |
| `parameter`                        | `NT_PARAMETER`                |
| `procedureDefinition`              | `NT_PROCEDURE_DEFINITION`      |
| `instruction`                      | `NT_INSTRUCTION`              |
| `architectureSpecificInstructionBody` | `NT_INSTRUCTION_BODY` |
| `x86InstructionBody`               | `NT_INSTRUCTION_BODY`             |
| `armInstructionBody`               | `NT_INSTRUCTION_BODY`             |
| `gpuInstructionBody`               | `NT_INSTRUCTION_BODY`             |
| `x86ShorthandInstruction`          | `NT_SHORTHAND_INSTRUCTION`      |
| `armShorthandInstruction`          | `NT_SHORTHAND_INSTRUCTION`      |
| `gpuShorthandInstruction`          | `NT_SHORTHAND_INSTRUCTION`      |
| `x86Mnemonic`                      | `NT_MNEMONIC`                 |
| `armMnemonic`                      | `NT_MNEMONIC`                 |
| `gpuMnemonic`                      | `NT_MNEMONIC`                 |
| `namespaceQualifier`               | `NT_NAMESPACE_QUALIFIER`       |
| `x86InstructionPrefix`             | `NT_INSTRUCTION_PREFIX`        |
| `armInstructionPrefix`             | `NT_ARM_INSTRUCTION_PREFIX`        |
| `gpuInstructionPrefix`             | `NT_INSTRUCTION_PREFIX` |
| `x86RepeatPrefix`                  | `NT_REPEAT_PREFIX_KIND`        |
| `x86SegmentPrefix`                 | `NT_SEGMENT_PREFIX_KIND`       |
| `x86AddressPrefix`                 | `NT_ADDRESS_PREFIX_KIND`       |
| `x86DataPrefix`                    | `NT_DATA_PREFIX_KIND`          |
| `x86VectorPrefix`                  | `NT_VECTOR_PREFIX_KIND`        |
| `x86OtherPrefix`                   | `NT_OTHER_PREFIX_KIND`         |
| `armConditionCode`                 | `NT_ARM_CONDITION_CODE`         |
| `armHintPrefix`                    | `NT_ARM_HINT_PREFIX`            |
| `x86InstructionMnemonic`           | `NT_MNEMONIC_...` (See Specific Mnemonics below) |
| `armInstructionMnemonic`           | `NT_MNEMONIC_...` (See ARM Specific Mnemonics below) |
| `gpuInstructionMnemonic`           | `NT_MNEMONIC_...` (See GPU Specific Mnemonics below) |
| `x86OperandList`                   | `NT_OPERAND_LIST`             |
| `armOperandList`                   | `NT_ARM_OPERAND_LIST`             |
| `gpuOperandList`                   | `NT_GPU_OPERAND_LIST`             |
| `x86Operand`                       | `NT_OPERAND`                  |
| `armOperand`                       | `NT_ARM_OPERAND`                  |
| `gpuOperand`                       | `NT_GPU_OPERAND`                  |
| `x86OperandSizeOverride`           | `NT_OPERAND_SIZE_OVERRIDE`     |
| `armOperandSizeOverride`           | `NT_ARM_OPERAND_SIZE_OVERRIDE`     |
| `gpuOperandSizeOverride`           | `NT_GPU_OPERAND_SIZE_OVERRIDE`     |
| `x86OperandType`                   | `NT_OPERAND_TYPE`             |
| `x86OperandKind`                   | `NT_OPERAND_KIND`             |
| `armOperandKind`                   | `NT_ARM_OPERAND_KIND`             |
| `gpuOperandKind`                   | `NT_GPU_OPERAND_KIND`             |
| `x86ModifiableOperand`            | `NT_MODIFIABLE_OPERAND`        |
| `armModifiableOperand`            | `NT_ARM_MODIFIABLE_OPERAND`        |
| `gpuModifiableOperand`            | `NT_GPU_MODIFIABLE_OPERAND`        |
| `x86RegisterOperand`              | `NT_REGISTER_OPERAND`         |
| `armRegister`                      | `NT_ARM_REGISTER_KIND`        |
| `gpuRegister`                      | `NT_GPU_REGISTER_KIND`        |
| `x86MemoryOperand`                 | `NT_MEMORY_OPERAND`           |
| `armMemoryOperand`                 | `NT_ARM_MEMORY_OPERAND`           |
| `gpuMemoryOperand`                 | `NT_GPU_MEMORY_OPERAND`           |
| `x86AddressBase`                   | `NT_ADDRESS_BASE`             |
| `armAddressBase`                   | `NT_ARM_ADDRESS_BASE`             |
| `gpuAddressBase`                   | `NT_ADDRESS_BASE`             |
| `x86AddressOffset`                 | `NT_ADDRESS_OFFSET`           |
| `armAddressOffset`                 | `NT_ARM_ADDRESS_OFFSET`           |
| `gpuAddressOffset`                 | `NT_ADDRESS_OFFSET`           |
| `x86AddressDisplacement`           | `NT_ADDRESS_DISPLACEMENT`     |
| `armAddressDisplacement`           | `NT_ARM_ADDRESS_DISPLACEMENT`     |
| `x86AddressScaleIndex`             | `NT_ADDRESS_SCALE_INDEX`       |
| `armAddressScaleIndex`             | `NT_ARM_ADDRESS_SCALE_INDEX`       |
| `armShiftedRegister`               | `NT_ARM_SHIFTED_REGISTER`     |
| `shiftType`                        | `NT_SHIFT_TYPE`             |
| `shiftOperation`                   | `NT_SHIFT_OPERATION`         |
| `x86AddressTerm`                   | `NT_ADDRESS_TERM`             |
| `armAddressTerm`                   | `NT_ADDRESS_TERM`             |
| `x86ScaleFactor`                   | `NT_SCALE_FACTOR`             |
| `x86Register`                      | `NT_REGISTER`                 |
| `armRegister`                      | `NT_REGISTER`                 |
| `gpuRegister`                      | `NT_REGISTER`                 |
| `generalRegister`                  | `NT_GENERAL_REGISTER_KIND`    |
| `segmentRegister`                  | `NT_SEGMENT_REGISTER_KIND`    |
| `controlRegister`                  | `NT_CONTROL_REGISTER_KIND`    |
| `debugRegister`                    | `NT_DEBUG_REGISTER_KIND`      |
| `mmxRegister`                      | `NT_MMX_REGISTER_KIND`        |
| `xmmRegister`                      | `NT_XMM_REGISTER_KIND`        |
| `ymmRegister`                      | `NT_YMM_REGISTER_KIND`        |
| `zmmRegister`                      | `NT_ZMM_REGISTER_KIND`        |
| `directive`                        | `NT_DIRECTIVE`                |
| `directiveName`                    | `NT_DIRECTIVE_NAME`            |
| `directiveArgumentList`            | `NT_DIRECTIVE_ARGUMENTS`        |
| `dataDirective`                    | `NT_DIRECTIVE_DATA`           |
| `equateDirective`                  | `NT_DIRECTIVE_EQUATE`         |
| `constDirective`                   | `NT_DIRECTIVE_CONST`          |
| `incbinDirective`                  | `NT_DIRECTIVE_INCBIN`         |
| `timesDirective`                   | `NT_DIRECTIVE_TIMES`          |
| `segmentDirective`                 | `NT_DIRECTIVE_SEGMENT`        |
| `useDirective`                     | `NT_DIRECTIVE_USE`            |
| `typeDirective`                    | `NT_DIRECTIVE_TYPE`           |
| `mutexDirective`                   | `NT_DIRECTIVE_MUTEX`          |
| `conditionDirective`               | `NT_DIRECTIVE_CONDITION`      |
| `globalDirective`                  | `NT_DIRECTIVE_GLOBAL`         |
| `externDirective`                  | `NT_DIRECTIVE_EXTERN`         |
| `alignDirective`                   | `NT_DIRECTIVE_ALIGN`          |
| `sectionDirective`                 | `NT_DIRECTIVE_SECTION`       |
| `ifDirective`                      | `NT_DIRECTIVE_IF`             |
|  `elifDirective`                   | `NT_DIRECTIVE_ELIF`             |
| `elseDirective`                    | `NT_DIRECTIVE_ELSE`             |
| `endifDirective`                   | `NT_DIRECTIVE_ENDIF`             |
| `ifdefDirective`                   | `NT_DIRECTIVE_IFDEF`             |
| `ifndefDirective`                  | `NT_DIRECTIVE_IFNDEF`             |
| `elifdefDirective`                 | `NT_DIRECTIVE_ELIFDEF`             |
| `elifndefDirective`                | `NT_DIRECTIVE_ELIFNDEF`             |
| `entryPointDirective`              | `NT_DIRECTIVE_ENTRYPOINT`     |
| `callingConventionDirective`       | `NT_DIRECTIVE_CALLINGCONVENTION` |
| `acpiDirective`                    | `NT_DIRECTIVE_ACPI`           |
| `ioDirective`                      | `NT_DIRECTIVE_IO`             |
| `structDirective`                  | `NT_DIRECTIVE_STRUCT_DIRECTIVE` |
| `cpuDirective`                     | `NT_DIRECTIVE_CPU`            |
| `bitsDirective`                    | `NT_DIRECTIVE_BITS`           |
| `stackDirective`                   | `NT_DIRECTIVE_STACK`          |
| `warningDirective`                 | `NT_DIRECTIVE_WARNING`        |
| `errorDirective`                   | `NT_DIRECTIVE_ERROR`          |
| `includeDirective`                 | `NT_DIRECTIVE_INCLUDE`        |
| `includeOnceDirective`             | `NT_DIRECTIVE_INCLUDEONCE`    |
| `listDirective`                    | `NT_DIRECTIVE_LIST`           |
| `nolistDirective`                  | `NT_DIRECTIVE_NOLIST`         |
| `debugDirective`                   | `NT_DIRECTIVE_DEBUG`          |
| `orgDirective`                     | `NT_DIRECTIVE_ORG`            |
| `mapDirective`                     | `NT_DIRECTIVE_MAP`            |
| `argDirective`                     | `NT_DIRECTIVE_ARG`            |
| `localDirective`                   | `NT_DIRECTIVE_LOCAL`          |
| `setDirective`                     | `NT_DIRECTIVE_SET`            |
| `unsetDirective`                   | `NT_DIRECTIVE_UNSET`          |
| `assertDirective`                  | `NT_DIRECTIVE_ASSERT`         |
| `optDirective`                     | `NT_DIRECTIVE_OPT`            |
| `evalDirective`                    | `NT_DIRECTIVE_EVAL`           |
| `repDirective`                     | `NT_DIRECTIVE_REP`            |
| `defaultDirective`                 | `NT_DIRECTIVE_DEFAULT`        |
| `exportDirective`                  | `NT_DIRECTIVE_EXPORT`         |
| `commonDirective`                  | `NT_DIRECTIVE_COMMON`         |
| `fileDirective`                    | `NT_DIRECTIVE_FILE`           |
| `lineDirective`                    | `NT_DIRECTIVE_LINE`           |
| `contextDirective`                 | `NT_DIRECTIVE_CONTEXT`        |
| `endcontextDirective`              | `NT_DIRECTIVE_ENDCONTEXT`     |
| `allocDirective`                   | `NT_DIRECTIVE_ALLOC`          |
| `freeDirective`                    | `NT_DIRECTIVE_FREE`           |
| `bitfieldDirective`                | `NT_DIRECTIVE_BITFIELD_DIRECTIVE` |
| `gpuDirective`                     | `NT_DIRECTIVE_GPU`            |
| `uefiDirective`                    | `NT_DIRECTIVE_UEFI`           |
| `staticDirective`                  | `NT_DIRECTIVE_STATIC`         |
| `dataBlockDirective`               | `NT_DIRECTIVE_DATABLOCK`      |
| `gdtDirective`                     | `NT_DIRECTIVE_GDT`            |
| `idtDirective`                     | `NT_DIRECTIVE_IDT`            |
| `linkerDirective`                  | `NT_DIRECTIVE_LINKER`         |
| `armDirective`                     | `NT_ARM_DIRECTIVE`            |
| `armDirectiveName`                 | `NT_ARM_DIRECTIVE_NAME`        |
| `armDirectiveArgumentList`         | `NT_ARM_DIRECTIVE_ARGUMENTS`    |
| `armDirectiveSyntax`               | `NT_ARM_DIRECTIVE_SYNTAX`       |
| `armDirectiveArch`                 | `NT_ARM_DIRECTIVE_ARCH`       |
| `armDirectiveThumb`                | `NT_ARM_DIRECTIVE_THUMB`       |
| `armDirectiveArm`                  | `NT_ARM_DIRECTIVE_ARM`       |
| `armDirectiveGlobal`               | `NT_ARM_DIRECTIVE_GLOBAL`       |
| `armDirectiveFunc`                 | `NT_ARM_DIRECTIVE_FUNC`       |
| `armDirectiveEndFunc`              | `NT_ARM_DIRECTIVE_ENDFUNC`       |
| `armDirectiveType`                 | `NT_ARM_DIRECTIVE_TYPE`       |
| `armDirectiveSize`                 | `NT_ARM_DIRECTIVE_SIZE`       |
| `linkerDirective`                  | `NT_LINKER_DIRECTIVE`         |
| `gdtDirective`                     | `NT_GDT_DIRECTIVE`            |
| `idtDirective`                     | `NT_IDT_DIRECTIVE`            |
| `gpuDirective`                     | `NT_GPU_DIRECTIVE`            |
| `uefiDirective`                    | `NT_UEFI_DIRECTIVE`           |
| `staticDirective`                  | `NT_STATIC_DIRECTIVE`         |
| `sectionSpecifier`                 | `NT_SECTION_SPECIFIER`        |
| `dataDefinition`                   | `NT_DATA_DEFINITION`          |
| `dataBlockDirective`               | `NT_DATA_BLOCK_DIRECTIVE`      |
| `dataBlockItem`                    | `NT_DATA_BLOCK_ITEM`           |
| `directiveArgument`                | `NT_DIRECTIVE_ARGUMENT`       |
| `directiveArgumentList`            | `NT_DIRECTIVE_ARGUMENT_LIST`   |
| `dataDirectiveKind`                | `NT_DATA_DIRECTIVE_KIND`       |
| `dataList`                         | `NT_DATA_LIST`                |
| `dataValue`                        | `NT_DATA_VALUE`               |
| `typeDirective`                    | `NT_TYPE_DIRECTIVE`           |
| `typeDefinition`                   | `NT_TYPE_DEFINITION`          |
| `basicType`                        | `NT_BASIC_TYPE`               |
| `arrayType`                        | `NT_ARRAY_TYPE`               |
| `pointerType`                      | `NT_POINTER_TYPE`             |
| `typeReference`                    | `NT_TYPE_REFERENCE`           |
| `structReference`                  | `NT_STRUCT_REFERENCE`         |
| `enumReference`                    | `NT_ENUM_REFERENCE`           |
| `templateReference`                | `NT_TEMPLATE_REFERENCE`       |
| `typeReferenceList`                | `NT_TYPE_REFERENCE_LIST`       |
| `enumDefinition`                   | `NT_ENUM_DEFINITION`          |
| `enumMemberList`                   | `NT_ENUM_MEMBER_LIST`          |
| `enumMember`                       | `NT_ENUM_MEMBER`              |
| `structDefinition`                 | `NT_STRUCT_DEFINITION`        |
| `structMemberList`                 | `NT_STRUCT_MEMBER_LIST`        |
| `structMember`                     | `NT_STRUCT_MEMBER`            |
| `bitfieldDirective`                | `NT_BITFIELD_DIRECTIVE`       |
| `bitfieldMemberList`               | `NT_BITFIELD_MEMBER_LIST`      |
| `bitfieldMember`                   | `NT_BITFIELD_MEMBER`          |
| `attributeList`                    | `NT_ATTRIBUTE_LIST`           |
| `attribute`                        | `NT_ATTRIBUTE`               |
| `constExpressionList`              | `NT_CONST_EXPRESSION_LIST`     |
| `armMacroDefinition`               | `NT_ARM_MACRO_DEFINITION`      |
| `armParameterList`                 | `NT_ARM_PARAMETER_LIST`        |
| `armParameter`                     | `NT_ARM_PARAMETER`            |
| `macroDefinition`                  | `NT_MACRO_DEFINITION`         |
| `moduleDefinition`                 | `NT_MODULE_DEFINITION`        |
| `registerClassDefinition`          | `NT_REGISTER_CLASS_DEFINITION` |
| `registerList`                     | `NT_REGISTER_LIST`            |
| `templateDefinition`               | `NT_TEMPLATE_DEFINITION`      |
| `templateParameterList`            | `NT_TEMPLATE_PARAMETER_LIST`   |
| `templateParameter`                | `NT_TEMPLATE_PARAMETER`       |
| `requiresClause`                   | `NT_REQUIRES_CLAUSE`          |
| `conceptConjunction`               | `NT_CONCEPT_CONJUNCTION`      |
| `conceptDisjunction`               | `NT_CONCEPT_DISJUNCTION`      |
| `conceptReference`                 | `NT_CONCEPT_REFERENCE`        |
| `templateElement`                  | `NT_TEMPLATE_ELEMENT`         |
| `unsafeBlock`                      | `NT_UNSAFE_BLOCK`             |
| `staticBlock`                      | `NT_STATIC_BLOCK`             |
| `templateCall`                     | `NT_TEMPLATE_CALL`            |
| `templateArgumentList`             | `NT_TEMPLATE_ARGUMENT_LIST`    |
| `templateArgument`                 | `NT_TEMPLATE_ARGUMENT`        |
| `comment`                          | `NT_COMMENT`                 |
| `label`                            | `NT_LABEL`                   |
| `shorthandOperator`                | `NT_SHORTHAND_OPERATOR_KIND`   |
| `threadCreation`                   | `NT_THREAD_CREATION`          |
| `expressionList`                   | `NT_EXPRESSION_LIST`          |
| `threadDirective`                  | `NT_THREAD_DIRECTIVE`          |
| `threadJoinDirective`              | `NT_THREAD_JOIN_DIRECTIVE`     |
| `threadTerminateDirective`         | `NT_THREAD_TERMINATE_DIRECTIVE` |
| `threadSleepDirective`             | `NT_THREAD_SLEEP_DIRECTIVE`    |
| `threadYieldDirective`             | `NT_THREAD_YIELD_DIRECTIVE`    |
| `threadLocalDirective`             | `NT_THREAD_LOCAL_DIRECTIVE`    |
| `operandList`                      | `NT_OPERAND_LIST`             |
| `operand`                          | `NT_OPERAND`                 |
| `operandSizeOverride`              | `NT_OPERAND_SIZE_OVERRIDE`     |
| `operandType`                      | `NT_OPERAND_TYPE`             |
| `operandKind`                      | `NT_OPERAND_KIND`             |
| `modifiableOperand`                | `NT_MODIFIABLE_OPERAND`        |
| `immediate`                        | `NT_IMMEDIATE`                |
| `registerOperand`                  | `NT_REGISTER_OPERAND`         |
| `memoryOperand`                    | `NT_MEMORY_OPERAND`           |
| `register`                         | `NT_REGISTER`                |
| `constant`                         | `NT_CONSTANT`                |
| `number`                           | `NT_NUMBER`                  |
| `hexNumber`                        | `NT_HEX_NUMBER`               |
| `binNumber`                        | `NT_BIN_NUMBER`               |
| `floatNumber`                      | `NT_FLOAT_NUMBER`             |
| `character`                        | `NT_CHARACTER`               |
| `escapeSequence`                   | `NT_ESCAPE_SEQUENCE`          |
| `addressLiteral`                   | `NT_ADDRESS_LITERAL`          |
| `expression`                       | `NT_EXPRESSION`               |
| `conditionalExpression`            | `NT_CONDITIONAL_EXPRESSION`   |
| `logicalOrExpression`              | `NT_LOGICAL_OR_EXPRESSION`     |
| `logicalAndExpression`             | `NT_LOGICAL_AND_EXPRESSION`    |
| `bitwiseOrExpression`              | `NT_BITWISE_OR_EXPRESSION`     |
| `bitwiseXorExpression`             | `NT_BITWISE_XOR_EXPRESSION`    |
| `bitwiseAndExpression`             | `NT_BITWISE_AND_EXPRESSION`    |
| `shiftExpression`                  | `NT_SHIFT_EXPRESSION`         |
| `additiveExpression`               | `NT_ADDITIVE_EXPRESSION`      |
| `multiplicativeExpression`         | `NT_MULTIPLICATIVE_EXPRESSION`|
| `unaryExpression`                  | `NT_UNARY_EXPRESSION`         |
| `typeConversion`                   | `NT_TYPE_CONVERSION`          |
| `sizeOfExpression`                 | `NT_SIZEOF_EXPRESSION`        |
| `alignOfExpression`                | `NT_ALIGNOF_EXPRESSION`       |
| `memoryAddress`                    | `NT_MEMORY_ADDRESS`           |
| `addressBase`                      | `NT_ADDRESS_BASE`             |
| `addressOffset`                    | `NT_ADDRESS_OFFSET`           |
| `addressDisplacement`              | `NT_ADDRESS_DISPLACEMENT`     |
| `addressScaleIndex`                | `NT_ADDRESS_SCALE_INDEX`       |
| `addressTerm`                      | `NT_ADDRESS_TERM`             |
| `scaleFactor`                      | `NT_SCALE_FACTOR`             |
| `stringLiteral`                    | `NT_STRING_LITERAL`           |
| `identifier`                       | `NT_IDENTIFIER`               |
| `digit`                            | `NT_DIGIT`                   |
| `hexDigit`                         | `NT_HEX_DIGIT`                |
| `binDigit`                         | `NT_BIN_DIGIT`                |
| `eof`                              | `NT_EOF`                     |
| `gpuAddress`                       | `NT_GPU_ADDRESS`              |
| `gpuAddressSpace`                  | `NT_GPU_ADDRESS_SPACE`         |
| `gpuAddressExpression`             | `NT_GPU_ADDRESS_EXPRESSION`    |
| `armAddress`                       | `NT_ARM_ADDRESS`              |
| `armInstruction`                   | `NT_INSTRUCTION`              |
| `gpuInstruction`                   | `NT_INSTRUCTION`              |
| `armInstructionPrefix`             | `NT_ARM_INSTRUCTION_PREFIX`    |
| `x86InstructionPrefix`             | `NT_INSTRUCTION_PREFIX`       |

**Specific Mnemonics:**

| Grammar Non-terminal      | NodeType              |
| :------------------------ | :------------------- |
| `mov`                     | `NT_MNEMONIC_MOV`     |
| `add`                     | `NT_MNEMONIC_ADD`     |
| `sub`                     | `NT_MNEMONIC_SUB`     |
| `jmp`                     | `NT_MNEMONIC_JMP`     |
| `call`                    | `NT_MNEMONIC_CALL`    |
| `ret`                     | `NT_MNEMONIC_RET`     |
| `push`                    | `NT_MNEMONIC_PUSH`    |
| `pop`                     | `NT_MNEMONIC_POP`     |
| `lea`                     | `NT_MNEMONIC_LEA`     |
| `cmp`                     | `NT_MNEMONIC_CMP`     |
| `test`                    | `NT_MNEMONIC_TEST`    |
| `and`                     | `NT_MNEMONIC_AND`     |
| `or`                      | `NT_MNEMONIC_OR`      |
| `xor`                     | `NT_MNEMONIC_XOR`     |
| `not`                     | `NT_MNEMONIC_NOT`     |
| `neg`                     | `NT_MNEMONIC_NEG`     |
| `mul`                     | `NT_MNEMONIC_MUL`     |
| `imul`                    | `NT_MNEMONIC_IMUL`    |
| `div`                     | `NT_MNEMONIC_DIV`     |
| `idiv`                    | `NT_MNEMONIC_IDIV`    |
| `shl`                     | `NT_MNEMONIC_SHL`     |
| `shr`                     | `NT_MNEMONIC_SHR`     |
| `sar`                     | `NT_MNEMONIC_SAR`     |
| `rol`                     | `NT_MNEMONIC_ROL`     |
| `ror`                     | `NT_MNEMONIC_ROR`     |
| `rcl`                     | `NT_MNEMONIC_RCL`     |
| `rcr`                     | `NT_MNEMONIC_RCR`     |
| `jz`                      | `NT_MNEMONIC_JZ`      |
| `jnz`                     | `NT_MNEMONIC_JNZ`     |
| `je`                      | `NT_MNEMONIC_JE`      |
| `jne`                     | `NT_MNEMONIC_JNE`     |
| `js`                      | `NT_MNEMONIC_JS`      |
| `jns`                     | `NT_MNEMONIC_JNS`     |
| `jg`                      | `NT_MNEMONIC_JG`      |
| `jge`                     | `NT_MNEMONIC_JGE`     |
| `jl`                      | `NT_MNEMONIC_JL`      |
| `jle`                     | `NT_MNEMONIC_JLE`     |
| `ja`                      | `NT_MNEMONIC_JA`      |
| `jae`                     | `NT_MNEMONIC_JAE`     |
| `jb`                      | `NT_MNEMONIC_JB`      |
| `jbe`                     | `NT_MNEMONIC_JBE`     |

**ARM-Specific Mnemonics:**

| Grammar Non-terminal      | NodeType              |
| :------------------------ | :------------------- |
| `mov`                     | `NT_MNEMONIC_MOV`     |
| `add`                     | `NT_MNEMONIC_ADD`     |
| `ldr`                     | `NT_MNEMONIC_LDR` |
| `str`                     | `NT_MNEMONIC_STR` |
| `b`                       | `NT_MNEMONIC_B` |
| `bl`                      | `NT_MNEMONIC_BL` |
| `cmp`                     | `NT_MNEMONIC_CMP`     |
| `tst`                     | `NT_MNEMONIC_TST` |
| `and`                     | `NT_MNEMONIC_AND`     |
| `orr`                     | `NT_MNEMONIC_ORR` |
| `eor`                     | `NT_MNEMONIC_EOR` |
| `sub`                     | `NT_MNEMONIC_SUB`     |
| `rsb`                     | `NT_MNEMONIC_RSB` |
| `mul`                     | `NT_MNEMONIC_MUL`     |
| `mla`                     | `NT_MNEMONIC_MLA` |
| `sdiv`                    | `NT_MNEMONIC_SDIV` |
| `udiv`                    | `NT_MNEMONIC_UDIV` |
| `push`                    | `NT_MNEMONIC_PUSH`    |
| `pop`                     | `NT_MNEMONIC_POP`     |

**GPU-Specific Mnemonics (Example CUDA PTX-like):**

| Grammar Non-terminal      | NodeType              |
| :------------------------ | :------------------- |
| `mov.b32`                 | `NT_MNEMONIC_MOV`     |
| `mov.b64`                 | `NT_MNEMONIC_MOV`     |
| `mov.f32`                 | `NT_MNEMONIC_MOV`     |
| `mov.f64`                 | `NT_MNEMONIC_MOV`     |
| `ld.global.f32`           | `NT_MNEMONIC_LD_GLOBAL` |
| `ld.global.b32`           | `NT_MNEMONIC_LD_GLOBAL` |
| `st.global.b32`           | `NT_MNEMONIC_ST_GLOBAL` |
| `st.global.f32`           | `NT_MNEMONIC_ST_GLOBAL` |
| `add.s32`                 | `NT_MNEMONIC_ADD`     |
| `sub.f32`                 | `NT_MNEMONIC_SUB`     |
| `mul.f32`                 | `NT_MNEMONIC_MUL`     |
| `mad.f32`                 | `NT_MNEMONIC_MAD`     |
| `setp.eq.s32`             | `NT_MNEMONIC_SETP`    |
| `bra`                     | `NT_MNEMONIC_BRA`     |
| `bra.uni`                 | `NT_MNEMONIC_BRA_UNI` |

