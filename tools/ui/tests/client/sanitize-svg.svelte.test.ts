import { sanitizeSvg } from '$lib/utils/sanitize-svg';
import { describe, expect, it } from 'vitest';

describe('SVG animation sanitization', () => {
	it('retains safe geometry animation and fragment use', () => {
		const clean = sanitizeSvg('<svg><defs><circle id="dot" r="2"/></defs><use href="#dot"/><animate attributeName="opacity" from="0" to="1" dur="1s"/></svg>');
		expect(clean).toContain('href="#dot"');
		expect(clean).toContain('attributeName="opacity"');
	});
	it.each(['href', 'xlink:href', 'style', 'onload', 'xml:base'])('neutralizes animation of %s', (target) => {
		const clean = sanitizeSvg(`<svg><set attributeName="${target}" to="javascript:alert(1)"/></svg>`);
		expect(clean.toLowerCase()).not.toContain('attributename=');
		expect(clean).not.toContain('javascript:');
	});
	it('removes unsafe static use URLs and animated paint URLs', () => {
		const clean = sanitizeSvg('<svg><use href="javascript:alert(1)"/><animate attributeName="fill" values="red;url(https://example.test/a)"/></svg>');
		expect(clean).not.toContain('javascript:');
		expect(clean).not.toContain('url(');
	});
});
