export const MIN_ZOOM = 1;
export const MAX_ZOOM = 8;
export const DOUBLE_CLICK_ZOOM = 2.5;
export const ZOOM_STEP_FACTOR = 1.35;

export interface ViewTransform {
  scale: number;
  x: number;
  y: number;
}

export interface Size2D {
  width: number;
  height: number;
}

export const IDENTITY_TRANSFORM: ViewTransform = { scale: 1, x: 0, y: 0 };

export function clampScale(scale: number): number {
  if (!Number.isFinite(scale) || scale <= MIN_ZOOM + 0.01) return MIN_ZOOM;
  return Math.min(MAX_ZOOM, scale);
}

/**
 * Clamp pan offsets so every corner of the zoomed image can be brought to the
 * exact center of the viewport for inspection/zooming, while preventing the
 * image from ever being dragged completely off-screen.
 */
export function clampPan(
  x: number,
  y: number,
  scale: number,
  baseSize: Size2D,
): { x: number; y: number } {
  const safeScale = clampScale(scale);
  if (safeScale <= MIN_ZOOM) return { x: 0, y: 0 };
  const maxX = Math.max(0, (Math.max(0, baseSize.width) * safeScale) / 2);
  const maxY = Math.max(0, (Math.max(0, baseSize.height) * safeScale) / 2);
  return {
    x: Math.max(-maxX, Math.min(maxX, x)),
    y: Math.max(-maxY, Math.min(maxY, y)),
  };
}

/**
 * Zoom to `targetScale` while keeping the image point currently located at
 * `(anchorX, anchorY)` (relative to viewport center) stationary on screen.
 * Passing `(0, 0)` zooms straight into whatever detail is currently centered.
 */
export function zoomAtPoint(
  current: ViewTransform,
  targetScale: number,
  anchorX: number,
  anchorY: number,
  baseSize: Size2D,
): ViewTransform {
  const nextScale = clampScale(targetScale);
  if (nextScale <= MIN_ZOOM) return IDENTITY_TRANSFORM;
  const ratio = nextScale / Math.max(MIN_ZOOM, current.scale);
  const rawX = anchorX - (anchorX - current.x) * ratio;
  const rawY = anchorY - (anchorY - current.y) * ratio;
  const clamped = clampPan(rawX, rawY, nextScale, baseSize);
  return { scale: nextScale, x: clamped.x, y: clamped.y };
}

/**
 * Double-click helper: when at fit scale, zoom in and bring the clicked point
 * to the center of the viewport; when already zoomed, reset to fit-to-screen.
 */
export function toggleDoubleClickZoom(
  current: ViewTransform,
  clickX: number,
  clickY: number,
  baseSize: Size2D,
  targetScale = DOUBLE_CLICK_ZOOM,
): ViewTransform {
  if (current.scale > MIN_ZOOM + 0.01) return IDENTITY_TRANSFORM;
  const nextScale = clampScale(targetScale);
  const clamped = clampPan(-clickX * nextScale, -clickY * nextScale, nextScale, baseSize);
  return { scale: nextScale, x: clamped.x, y: clamped.y };
}

export function panBy(
  current: ViewTransform,
  dx: number,
  dy: number,
  baseSize: Size2D,
): ViewTransform {
  if (current.scale <= MIN_ZOOM) return IDENTITY_TRANSFORM;
  const clamped = clampPan(current.x + dx, current.y + dy, current.scale, baseSize);
  return { scale: current.scale, x: clamped.x, y: clamped.y };
}

/**
 * Scale factor needed so 1 CSS pixel equals 1 natural image pixel (1:1 view),
 * clamped to `[MIN_ZOOM, MAX_ZOOM]`. Returns null if image is already >= 1:1.
 */
export function computeNativeScale(natural: Size2D, fitted: Size2D): number | null {
  if (natural.width <= 0 || natural.height <= 0 || fitted.width <= 0 || fitted.height <= 0) {
    return null;
  }
  const ratio = Math.max(natural.width / fitted.width, natural.height / fitted.height);
  if (ratio <= 1.05) return null;
  return clampScale(ratio);
}
