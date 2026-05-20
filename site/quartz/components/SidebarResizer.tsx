// @ts-ignore
import sidebarResizerScript from "./scripts/sidebarResizer.inline"
import styles from "./styles/sidebarResizer.scss"
import { QuartzComponent, QuartzComponentConstructor } from "./types"

// SidebarResizer is invisible. It only exists to attach a beforeDOMLoaded
// script (which injects drag handles between the sidebars and the article)
// and to pull in the matching SCSS. Render it once per layout.
const SidebarResizer: QuartzComponent = () => null

SidebarResizer.beforeDOMLoaded = sidebarResizerScript
SidebarResizer.css = styles

export default (() => SidebarResizer) satisfies QuartzComponentConstructor
