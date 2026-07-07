/**
 * Validation and coercion of user-entered plugin parameter values.
 *
 * Every value is validated as a complete literal of its type
 *  before being coerced.
 */

import type { ImplementationInfo } from "../types/plugin";

/** Whole-string literal patterns per Python type name. */
const INT_RE = /^[+-]?\d+$/;
const FLOAT_RE = /^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$/;
const BOOL_RE = /^(true|false)$/i;

/**
 * Check that `value` is a complete literal of the registry `type`
 * ("int", "float", "bool"; anything else = free text).
 *
 * @returns An error message, or `null` when the value is valid.
 */
export function validateParamValue(value: string, type: string): string | null {
  const v = value.trim();
  switch (type) {
    case "int":
      return INT_RE.test(v) ? null : "must be a whole number";
    case "float":
      return FLOAT_RE.test(v)
        ? null
        : "must be a number (use '.' as the decimal separator)";
    case "bool":
      return BOOL_RE.test(v) ? null : "must be true or false";
    default:
      return null;
  }
}

/**
 * Coerce a validated string value to the native JS type the backend
 * expects. Call only after {@link validateParamValue} passed.
 */
export function coerceParamValue(value: string, pythonType: string): unknown {
  switch (pythonType) {
    case "int":
      return parseInt(value, 10);
    case "float":
      return parseFloat(value);
    case "bool":
      return value.trim().toLowerCase() === "true";
    default:
      return value;
  }
}

/** Result of {@link collectParams}: either the params dict or the first error. */
export type CollectParamsResult =
  | { params: Record<string, unknown> }
  | { error: string };

/**
 * Build the params dict for one step: registry defaults overridden by
 * the card's validated user input.
 *
 * @returns The coerced params, or the first validation error found
 *          (invalid literal, or an empty required parameter).
 */
export function collectParams(
  impl: ImplementationInfo,
  values: Record<string, string>
): CollectParamsResult {
  const params: Record<string, unknown> = {};
  for (const p of impl.params) {
    const userVal = values[p.name];
    if (userVal !== undefined && userVal !== "") {
      const error = validateParamValue(userVal, p.type);
      if (error) {
        return { error: `parameter "${p.name}" ${error} (got "${userVal}")` };
      }
      params[p.name] = coerceParamValue(userVal, p.type);
    } else if (p.default != null) {
      params[p.name] = p.default;
    } else if (p.required) {
      return { error: `parameter "${p.name}" is required` };
    }
  }
  return { params };
}
