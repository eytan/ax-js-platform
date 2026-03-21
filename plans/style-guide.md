# Style Guide: ax-js TypeScript Conventions

Coding conventions for ax-js. Based on Meta TypeScript conventions with
exemptions for mathematical code.

Cross-references: `eslint.config.mjs`, `.prettierrc`, `CLAUDE.md`

---

## 1. Guiding Principle: Incremental Adoption

- **New code** follows all rules in this document.
- **Existing code** is updated opportunistically — when touching a file for a logic
  change, fix style violations in the lines you modify, but do not reformat entire
  files in the same PR.
- **Formatting-only PRs** are welcome but must be separate from logic PRs.

---

## 2. Enforced Meta TS Conventions

These rules are enforced by ESLint and Prettier. Violations fail CI.

### Types

- **Use `Array<T>`**, not `T[]`. Enforced by `@typescript-eslint/array-type: generic`.
- **Use `import type`** for type-only imports. Enforced by
  `@typescript-eslint/consistent-type-imports`.
- **Explicit return types** on all functions. Enforced by
  `@typescript-eslint/explicit-function-return-type`.
- **Prefer `readonly`** on class properties that are never reassigned. Enforced by
  `@typescript-eslint/prefer-readonly`.

### Control flow

- **Use `for...of`** over `for...in` for iteration. Enforced by
  `@typescript-eslint/prefer-for-of`.
- **Strict equality** (`===`/`!==`). Enforced by `eqeqeq: always`.
- **Always use braces** for `if`/`else`/`for`/`while`. Enforced by `curly: all`.
- **Exhaustive switch** statements on discriminated unions. Enforced by
  `@typescript-eslint/switch-exhaustiveness-check`.

### Variables

- **`const` over `let`**. Never use `var`. Enforced by `prefer-const` and `no-var`.
- **No variable shadowing.** Enforced by `@typescript-eslint/no-shadow`.

### Formatting (Prettier)

- Semicolons: always.
- Quotes: double (`""`).
- Trailing commas: all.
- Print width: 100.
- Tab width: 2.
- Arrow parens: always (`(x) => ...`).

---

## 3. Math Code Exemptions

Mathematical code has different readability requirements than application code.
The following ESLint rules are OFF or relaxed project-wide to accommodate this:

### Allowed patterns

- **Single-letter variables**: `L`, `K`, `V`, `X`, `A`, `alpha`, `Sigma`, `Kstar`.
  These match standard linear algebra and GP notation. The ESLint naming convention
  rule explicitly allows `UPPER_CASE` and `PascalCase` for variables to support this.

- **Traditional `for` loops** in hot paths (kernel evaluation, matrix operations,
  Cholesky factorization). `unicorn/no-for-loop` is OFF because `for...of` on
  `Float64Array` has measurable overhead in tight numerical loops.

- **`Array.reduce`** for summations and accumulations.
  `unicorn/no-array-reduce` is OFF.

- **`Array.forEach`** where appropriate.
  `unicorn/no-array-for-each` is OFF.

- **Abbreviations** in mathematical contexts (`idx`, `len`, `dim`, `cov`).
  `unicorn/prevent-abbreviations` is OFF.

- **Math.floor instead of Math.trunc** where semantically equivalent.
  `unicorn/prefer-math-trunc` is OFF.

### Rationale

These exemptions exist because:
1. Mathematical code is read alongside papers and reference implementations (BoTorch,
   GPyTorch). Matching their notation reduces translation errors.
2. Performance matters in inner loops. A 2x slowdown in kernel evaluation multiplies
   across all predictions.
3. The codebase is small enough (~5k lines of math code) that these exemptions do not
   cause maintenance problems.

---

## 4. Naming Convention: The Hybrid Rule

ax-js uses a deliberate hybrid naming convention to minimize bugs at the
Python/TypeScript boundary.

### State and data interfaces (`types.ts`): snake_case

All interfaces that represent serialized state use **snake_case** property names,
matching the JSON wire format from Python:

```typescript
interface SingleTaskGPState {
  model_type: "SingleTaskGP";
  train_X: Array<Array<number>>;
  train_Y: Array<Array<number>>;
  mean_constant: number;
  active_dims?: Array<number>;
}
```

ESLint naming conventions exempt interface properties from camelCase enforcement.

### Runtime code: camelCase

All runtime variables, function parameters, and local variables use **camelCase**:

```typescript
const trainX = Matrix.from2D(state.train_X);
const meanConstant = state.mean_constant;
const inputTransform = buildInputTransform(state.input_transform);
```

### Conversion boundary

The snake_case-to-camelCase conversion happens at a single point: the constructor
or deserialization function. After that point, all code uses camelCase.

### Class names: aligned across languages

Class names match BoTorch/GPyTorch exactly: `SingleTaskGP`, `MaternKernel`,
`ScaleKernel`, `RBFKernel`, `KumaraswamyCDFTransform`. This makes it trivial to
find the corresponding Python implementation.

### Why this approach

- **snake_case in interfaces = grep-able across codebases.** Searching for
  `mean_constant` finds both the Python export and the TypeScript consumer.
- **Zero conversion bugs.** The wire format is never silently renamed; if a Python
  field changes name, the TypeScript interface breaks at compile time.
- **Clear boundary.** Developers always know which "side" of the boundary they are
  on by looking at the casing.

---

## 5. Performance Conventions

### Use Float64Array for numerical data

All numerical arrays use `Float64Array` (via the `Matrix` class), never
`Array<number>`. This ensures:
- Predictable memory layout (contiguous, no boxing).
- Consistent Float64 precision (no silent Float32 truncation).
- Cache-friendly access patterns for row-major matrix operations.

### Avoid unnecessary allocations in hot paths

- Kernel evaluation and matrix operations should not allocate temporary arrays
  per element. Pre-allocate result matrices.
- Use in-place operations on `Matrix` where the result dimensions match.

### Prefer in-place operations on Matrix

The `Matrix` class supports direct indexing via `.data[i * cols + j]`. In tight
loops, prefer direct indexing over method calls:

```typescript
// Preferred in hot paths
for (let i = 0; i < n; i++) {
  for (let j = 0; j < m; j++) {
    result.data[i * m + j] = a.data[i * k + p] * b.data[p * m + j];
  }
}
```

---

## 6. ESLint Config Rationale

| Rule | Setting | Reason |
|------|---------|--------|
| `no-explicit-any` | OFF | Incremental adoption. New code should avoid `any`, but legacy code has justified uses. |
| `unicorn/no-for-loop` | OFF | Traditional `for` loops needed for Float64Array hot paths. |
| `unicorn/no-array-reduce` | OFF | `reduce` is idiomatic for summation/accumulation in math code. |
| `unicorn/no-array-for-each` | OFF | `forEach` is acceptable where no return value is needed. |
| `unicorn/prevent-abbreviations` | OFF | Math abbreviations (`idx`, `dim`, `cov`) are standard notation. |
| `unicorn/prefer-math-trunc` | OFF | `Math.floor` and `Math.trunc` are equivalent for non-negative values. |
| `consistent-type-imports` | ON | Prevents runtime import of type-only dependencies. Reduces bundle size. |
| `explicit-function-return-type` | ON | Makes function contracts clear. Critical for a numerical library. |
| `prefer-readonly` | ON | Prevents accidental mutation of model state. |
| `switch-exhaustiveness-check` | ON | Catches unhandled discriminated union variants (model_type, kernel_type). |
| `array-type: generic` | ON | `Array<T>` is more readable than `T[]` for complex types like `Array<Array<number>>`. |
| `import ordering` | ON | Groups: type, builtin, external, internal, parent, sibling, index. |

---

## 7. Documentation Standards

### Comments explain WHY, not WHAT

```typescript
// BAD: Multiply alpha by kernel matrix
const pred = K.matmul(alpha);

// GOOD: Posterior mean = K(x*, X) @ alpha, where alpha = K^{-1}(y - m)
const pred = K.matmul(alpha);
```

### Formula references

When implementing a mathematical formula, include a reference to the source:

```typescript
// Matern 5/2: (1 + sqrt(5)*d + 5/3*d^2) * exp(-sqrt(5)*d)
// See Rasmussen & Williams (2006), Eq. 4.17
```

### TSDoc on public API only

- All exported functions and classes in `src/index.ts`, `src/acquisition/index.ts`,
  and `src/viz/index.ts` should have TSDoc comments.
- Internal/private functions do not need TSDoc unless the logic is non-obvious.
- TypeDoc generates API reference from these annotations (`npm run docs`).

### No redundant documentation

- Do not add inline comments that restate the code.
- Do not add README files or markdown documentation unless explicitly requested.
- The codebase should be self-documenting through clear naming and types.

---

## 8. File Organization

### One class per file

Each kernel, model, or transform gets its own file:
- `src/kernels/matern.ts` — `MaternKernel`
- `src/models/single_task.ts` — `SingleTaskGP`
- `src/transforms/normalize.ts` — `Normalize`

### types.ts for shared interfaces

Each module directory has a `types.ts` containing the interfaces and type aliases
used across that module:
- `src/kernels/types.ts` — `Kernel`, `KernelState`, kernel discriminated unions
- `src/models/types.ts` — `Model`, `AnyModelState`, model discriminated unions
- `src/viz/types.ts` — `RGB`, `ParamSpec`, `RenderPredictor`, option interfaces

### index.ts as pure barrel

`index.ts` files are re-export barrels only. They contain no logic, no side effects,
and no type definitions. This makes the public API surface explicit and auditable:

```typescript
// src/index.ts — pure re-exports
export { Predictor } from "./predictor";
export { loadModel } from "./io/deserialize";
export type { ExperimentState } from "./models/types";
```

### Test file naming

Test files mirror source files: `src/kernels/matern.ts` is tested by
`test/kernels/matern.test.ts`. Integration tests live in `test/integration/`.
