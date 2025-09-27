from typing import Dict, Any, List, Optional
import base64
from io import BytesIO
from PIL import Image

from app.models.schemas import ChatRequest, ChatResponse
from app.core.ai_models import ai_manager


class DesignCoach:
    """
    AI-powered conversational design coach
    """

    def __init__(self):
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}
        self.analysis_context: Dict[str, Any] = {}

    async def process_message(self, request: ChatRequest) -> ChatResponse:
        """
        Process user message and provide design coaching response
        """
        try:
            # Build context for the conversation
            context = self._build_context(request)

            # Generate response using VLM
            response_text = await self._generate_response(request.message, context, request.image_context)

            # Extract suggestions from response
            suggestions = self._extract_suggestions(response_text)

            # Find related analyses
            related_analyses = self._find_related_analyses(request.analysis_id)

            # Store conversation
            self._store_conversation(request.message, response_text, request.analysis_id)

            return ChatResponse(
                response=response_text,
                suggestions=suggestions,
                related_analyses=related_analyses
            )

        except Exception as e:
            return ChatResponse(
                response=f"I apologize, but I'm having trouble processing your request right now. Could you please try rephrasing your question? Error: {str(e)}",
                suggestions=[],
                related_analyses=[]
            )

    async def process_feedback(self, analysis_id: str, rating: int, comments: str = ""):
        """
        Process user feedback on analysis quality
        """
        # Store feedback for model improvement
        feedback_data = {
            "analysis_id": analysis_id,
            "rating": rating,
            "comments": comments,
            "timestamp": "now"  # In production, use proper timestamp
        }

        # In production, store in database for model training
        print(f"Feedback received: {feedback_data}")

    def _build_context(self, request: ChatRequest) -> str:
        """
        Build conversation context including analysis results
        """
        context_parts = []

        # Add analysis context if available
        if request.analysis_id and request.analysis_id in self.analysis_context:
            analysis = self.analysis_context[request.analysis_id]
            context_parts.append(f"Previous analysis results: {analysis}")

        # Add conversation history
        conversation_key = request.analysis_id or "general"
        if conversation_key in self.conversation_history:
            recent_history = self.conversation_history[conversation_key][-3:]  # Last 3 exchanges
            for exchange in recent_history:
                context_parts.append(f"User: {exchange['user']}")
                context_parts.append(f"Assistant: {exchange['assistant']}")

        return "\n".join(context_parts)

    async def _generate_response(self, message: str, context: str, image_context: Optional[str] = None) -> str:
        """
        Generate AI response using VLM
        """
        # Build comprehensive prompt
        system_prompt = """You are a professional design coach and art director with expertise in:
- Visual design principles and composition
- Color theory and psychology
- Typography and readability
- User experience and accessibility
- Brand consistency and marketing effectiveness
- Platform-specific design optimization

Provide helpful, actionable advice in a friendly but professional tone.
Be specific with recommendations and explain the reasoning behind your suggestions.
If technical details are needed, provide them clearly.
"""

        full_prompt = f"{system_prompt}\n\nContext: {context}\n\nUser question: {message}\n\nProvide a helpful and detailed response:"

        # If image context is provided, include it
        if image_context:
            try:
                # Decode base64 image
                image_data = base64.b64decode(image_context.split(',')[1] if ',' in image_context else image_context)
                image = Image.open(BytesIO(image_data))

                # Use VLM with image
                response = await ai_manager.analyze_design(image, full_prompt)
            except Exception:
                # Fallback to text-only response
                response = await self._generate_text_response(full_prompt)
        else:
            response = await self._generate_text_response(full_prompt)

        return self._format_response(response, message)

    async def _generate_text_response(self, prompt: str) -> str:
        """
        Generate text-only response for questions without images
        """
        # For text-only responses, we'll use predefined responses based on keywords
        # In production, integrate with a text LLM like GPT-3.5 or Claude

        message_lower = prompt.lower()

        if "color" in message_lower:
            return self._get_color_advice(prompt)
        elif "font" in message_lower or "typography" in message_lower:
            return self._get_typography_advice(prompt)
        elif "layout" in message_lower or "composition" in message_lower:
            return self._get_layout_advice(prompt)
        elif "mobile" in message_lower or "responsive" in message_lower:
            return self._get_mobile_advice(prompt)
        elif "accessibility" in message_lower or "contrast" in message_lower:
            return self._get_accessibility_advice(prompt)
        elif "brand" in message_lower or "consistency" in message_lower:
            return self._get_branding_advice(prompt)
        else:
            return self._get_general_advice(prompt)

    def _get_color_advice(self, prompt: str) -> str:
        """Provide color-related advice"""
        return """Great question about color! Here are some key principles:

**Color Harmony:**
- Use complementary colors (opposite on color wheel) for high impact
- Try analogous colors (adjacent on wheel) for calm, cohesive feels
- Triadic schemes (3 colors equally spaced) create vibrant yet balanced designs

**Practical Tips:**
- Limit your palette to 3-5 colors plus neutrals
- Use the 60-30-10 rule: 60% dominant, 30% secondary, 10% accent
- Test your colors for accessibility (4.5:1 contrast ratio minimum)

**Psychology:**
- Blue builds trust (great for corporate, tech)
- Red creates urgency (effective for CTAs)
- Green suggests growth, nature, money
- Purple conveys luxury, creativity

Would you like me to analyze a specific color scheme you're working with?"""

    def _get_typography_advice(self, prompt: str) -> str:
        """Provide typography advice"""
        return """Typography is crucial for both readability and brand personality!

**Font Pairing Guidelines:**
- Pair serif with sans-serif for classic contrast
- Use different weights of the same font family for harmony
- Limit to 2-3 font families maximum

**Hierarchy Best Practices:**
- Create clear size relationships (use ratios like 1.2x, 1.5x, 2x)
- Use contrast in weight (light, regular, bold)
- Maintain consistent spacing and alignment

**Readability Essentials:**
- Minimum 16px for body text on web
- Line height of 1.4-1.6 for comfortable reading
- Keep line length to 45-75 characters
- Ensure sufficient contrast with background

**Platform Considerations:**
- Mobile: Larger sizes, simpler fonts
- Print: Higher contrast, serif often works better
- Digital: Sans-serif typically more readable on screens

What specific typography challenge are you facing?"""

    def _get_layout_advice(self, prompt: str) -> str:
        """Provide layout and composition advice"""
        return """Layout is the foundation of effective design! Here's how to create strong compositions:

**Grid Systems:**
- Use 12-column grids for flexible layouts
- Establish consistent margins and gutters
- Align elements to grid lines for professional look

**Visual Hierarchy:**
- Size: Larger elements draw attention first
- Position: Top-left gets noticed first (in Western cultures)
- Contrast: High contrast creates focal points
- White space: Use it to guide the eye and create breathing room

**Composition Principles:**
- Rule of thirds: Place key elements at intersection points
- Golden ratio: Use 1.618 ratio for pleasing proportions
- Balance: Distribute visual weight evenly
- Proximity: Group related elements together

**White Space Management:**
- Aim for 20-40% white space in most designs
- Use white space to separate content sections
- Don't fear empty space - it improves comprehension

**Flow Patterns:**
- Z-pattern: For simple layouts (logo top-left, CTA bottom-right)
- F-pattern: For text-heavy content
- Circular: For keeping users engaged on single page

Need help with a specific layout challenge?"""

    def _get_mobile_advice(self, prompt: str) -> str:
        """Provide mobile optimization advice"""
        return """Mobile design requires special considerations for usability and performance:

**Touch Interface:**
- Minimum touch target size: 44px (iOS) or 48px (Android)
- Add spacing between interactive elements
- Make buttons thumb-friendly (bottom of screen easier to reach)

**Typography for Mobile:**
- Minimum 16px font size to prevent zoom
- Increase line height for easier reading
- Use larger font sizes for CTAs

**Layout Adaptations:**
- Single column layouts work best
- Stack elements vertically
- Prioritize content - show most important items first
- Reduce navigation complexity

**Performance Optimization:**
- Optimize images for different screen densities
- Use responsive images with srcset
- Minimize file sizes for faster loading
- Consider connection speeds

**Platform Guidelines:**
- Follow iOS Human Interface Guidelines
- Adhere to Material Design for Android
- Test on actual devices, not just browser resize

**Accessibility:**
- Ensure color contrast meets WCAG guidelines
- Support zoom up to 200%
- Test with screen readers

What specific mobile challenge are you trying to solve?"""

    def _get_accessibility_advice(self, prompt: str) -> str:
        """Provide accessibility advice"""
        return """Accessibility makes your design usable by everyone! Here's how to ensure inclusive design:

**Color and Contrast:**
- WCAG AA requires 4.5:1 contrast ratio for normal text
- 3:1 ratio for large text (18pt+ or 14pt+ bold)
- Don't rely on color alone to convey information
- Test with color blindness simulators

**Typography Accessibility:**
- Use readable fonts (avoid decorative fonts for body text)
- Minimum 16px font size
- Allow text zoom up to 200%
- Maintain adequate line spacing (1.5x minimum)

**Navigation and Interaction:**
- Provide clear focus indicators for keyboard navigation
- Ensure all interactive elements are keyboard accessible
- Use meaningful link text (not "click here")
- Provide skip links for main content

**Images and Media:**
- Add alt text for all informative images
- Use empty alt="" for decorative images
- Provide captions for videos
- Include transcripts for audio content

**Testing Methods:**
- Use screen reader software (NVDA, JAWS, VoiceOver)
- Navigate using only keyboard
- Test with users who have disabilities
- Use automated tools like axe or WAVE

**Benefits:**
- Larger audience reach
- Better SEO rankings
- Improved usability for everyone
- Legal compliance (ADA, Section 508)

Would you like specific guidance on testing your current design for accessibility?"""

    def _get_branding_advice(self, prompt: str) -> str:
        """Provide branding and consistency advice"""
        return """Brand consistency builds trust and recognition! Here's how to maintain it:

**Visual Identity Elements:**
- Logo usage: Maintain clear space, correct proportions
- Color palette: Document exact hex/RGB values
- Typography: Define primary and secondary fonts
- Imagery style: Consistent photography/illustration approach

**Brand Guidelines Documentation:**
- Create a style guide with all brand elements
- Include do's and don'ts for logo usage
- Specify color combinations and applications
- Document voice and tone guidelines

**Consistency Across Platforms:**
- Maintain core elements while adapting to platform constraints
- Scale appropriately for different screen sizes
- Adjust messaging tone for platform audience
- Keep visual hierarchy consistent

**Measuring Brand Consistency:**
- Conduct regular brand audits
- Check all touchpoints (website, social, print, packaging)
- Get feedback from customers about brand perception
- Use tools to monitor brand mentions and usage

**Evolution vs. Consistency:**
- Small refinements are okay
- Major changes need careful consideration
- Test changes with target audience
- Document all updates in brand guidelines

**Team Implementation:**
- Train all team members on brand guidelines
- Create templates for common materials
- Establish approval processes for brand materials
- Regular check-ins to ensure compliance

What aspect of brand consistency are you looking to improve?"""

    def _get_general_advice(self, prompt: str) -> str:
        """Provide general design advice"""
        return """I'm here to help with all aspects of design! Here are some fundamental principles that apply across all projects:

**Design Process:**
1. Research and understand your audience
2. Define clear objectives and constraints
3. Sketch and ideate multiple concepts
4. Create wireframes before visual design
5. Test with real users and iterate

**Universal Design Principles:**
- **Contrast:** Make important elements stand out
- **Repetition:** Create consistency and unity
- **Alignment:** Connect and organize elements
- **Proximity:** Group related items together

**Quality Checklist:**
- Does it serve the user's needs?
- Is the hierarchy clear?
- Are the colors accessible?
- Is it readable at different sizes?
- Does it work on mobile?
- Is it consistent with brand guidelines?

**Common Mistakes to Avoid:**
- Too many fonts or colors
- Poor contrast and readability
- Cluttered layouts without white space
- Ignoring mobile users
- Forgetting about accessibility

**Staying Current:**
- Follow design leaders on social media
- Study award-winning work (Awwwards, Behance, Dribbble)
- Read design blogs and publications
- Attend conferences and workshops
- Practice recreating designs you admire

**Tools and Resources:**
- Design: Figma, Sketch, Adobe Creative Suite
- Color: Coolors, Adobe Color, Contrast checkers
- Typography: Google Fonts, Typekit, FontPair
- Inspiration: Behance, Dribbble, Pinterest
- Learning: Coursera, Udemy, YouTube tutorials

What specific design challenge would you like help with? I can provide more targeted advice based on your project needs!"""

    def _format_response(self, response: str, original_message: str) -> str:
        """
        Format and personalize the response
        """
        # Clean up response and add personal touch
        formatted_response = response.strip()

        # Add contextual closing based on message type
        if "?" in original_message:
            formatted_response += "\n\nFeel free to ask follow-up questions - I'm here to help you create amazing designs! 🎨"
        else:
            formatted_response += "\n\nIs there anything specific about this you'd like me to elaborate on?"

        return formatted_response

    def _extract_suggestions(self, response: str) -> List[str]:
        """
        Extract actionable suggestions from response
        """
        suggestions = []

        # Look for bullet points or numbered lists
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if (line.startswith('- ') or
                line.startswith('• ') or
                any(line.startswith(f'{i}.') for i in range(1, 10))):

                # Clean up the suggestion
                suggestion = line.lstrip('- •0123456789. ').strip()
                if len(suggestion) > 10 and len(suggestion) < 100:  # Reasonable length
                    suggestions.append(suggestion)

        return suggestions[:5]  # Limit to 5 suggestions

    def _find_related_analyses(self, analysis_id: Optional[str]) -> List[str]:
        """
        Find related analysis IDs
        """
        # In production, search database for related analyses
        # For now, return empty list
        return []

    def _store_conversation(self, user_message: str, assistant_response: str, analysis_id: Optional[str]):
        """
        Store conversation for context in future interactions
        """
        conversation_key = analysis_id or "general"

        if conversation_key not in self.conversation_history:
            self.conversation_history[conversation_key] = []

        self.conversation_history[conversation_key].append({
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": "now"  # In production, use proper timestamp
        })

        # Keep only last 10 exchanges
        if len(self.conversation_history[conversation_key]) > 10:
            self.conversation_history[conversation_key] = self.conversation_history[conversation_key][-10:]