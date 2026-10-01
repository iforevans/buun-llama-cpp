import { SVG } from '$lib/constants';
import DOMPurify from 'dompurify';

/**
 * SMIL values bypass ordinary attribute URI checks. Only permit geometry,
 * opacity and color targets, never URI/style/event targets. Paint servers are
 * excluded from animation values as well. Static SVG attributes still go
 * through DOMPurify's ordinary policy.
 */
const animationTargets = new Set([
	'x', 'y', 'x1', 'x2', 'y1', 'y2', 'cx', 'cy', 'r', 'rx', 'ry',
	'width', 'height', 'd', 'points', 'transform', 'opacity', 'fill-opacity',
	'stroke-opacity', 'stroke-width', 'stroke-dasharray', 'stroke-dashoffset',
	'fill', 'stroke', 'color', 'stop-color', 'stop-opacity'
]);

DOMPurify.addHook('uponSanitizeAttribute', (node, data) => {
	if (!['animate', 'set', 'animatetransform', 'animatemotion'].includes(node.nodeName.toLowerCase())) return;
	if (data.attrName === 'attributename' && !animationTargets.has(data.attrValue.trim().toLowerCase())) {
		data.keepAttr = false;
	}
	if (['to', 'from', 'by', 'values'].includes(data.attrName) && /url\s*\(|[\\:]/i.test(data.attrValue)) {
		data.keepAttr = false;
	}
});

/**
 * Sanitizes a raw svg string for safe inline rendering.
 * Returns the cleaned svg markup, or an empty string when the input is not a
 * usable svg, exceeds the size ceiling, or sanitizes to nothing. An empty
 * return tells the caller to keep the raw code block instead of rendering.
 */
export function sanitizeSvg(source: string): string {
	const trimmed = source.trim();

	if (!trimmed || trimmed.length > SVG.MAX_BYTES) return '';

	if (!trimmed.startsWith(SVG.TAG_PREFIX)) return '';

	const clean = DOMPurify.sanitize(trimmed, SVG.SANITIZE_CONFIG) as unknown as string;

	if (!clean || !clean.includes(SVG.TAG_PREFIX)) return '';

	return clean;
}
