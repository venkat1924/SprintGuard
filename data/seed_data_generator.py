"""
Generate realistic synthetic historical story data for SprintGuard PoC.
Creates 150 stories with patterns that reflect real-world Agile challenges.
"""
import json
import random
from pathlib import Path


# Story templates by risk level
HIGH_RISK_STORIES = [
    "Implement OAuth2 authentication for the REST API",
    "Migrate user database from MySQL to PostgreSQL",
    "Integrate third-party payment gateway (Stripe)",
    "Refactor legacy monolith to microservices architecture",
    "Build real-time WebSocket notification system",
    "Implement full-text search with Elasticsearch integration",
    "Set up CI/CD pipeline with automated deployment",
    "Add API rate limiting and security middleware",
    "Migrate legacy codebase from Python 2 to Python 3",
    "Implement distributed caching with Redis",
    "Build data export feature for GDPR compliance",
    "Integrate with third-party CRM via REST API",
    "Implement end-to-end encryption for messages",
    "Set up Kubernetes cluster for containerized deployment",
    "Build automated database backup and recovery system",
    "Implement SSO integration with enterprise LDAP",
    "Refactor database schema to improve query performance",
    "Build real-time analytics dashboard with streaming data",
    "Implement API versioning and backward compatibility",
    "Set up multi-region database replication",
]

MEDIUM_RISK_STORIES = [
    "Create user profile management page",
    "Implement email notification system",
    "Build admin dashboard for user management",
    "Add file upload functionality with validation",
    "Create responsive navigation menu",
    "Implement search functionality for products",
    "Build CSV export feature for reports",
    "Add pagination to data tables",
    "Create form validation with error messages",
    "Implement user role-based permissions",
    "Build notification preferences page",
    "Add filtering and sorting to list views",
    "Create settings page for user preferences",
    "Implement password reset functionality",
    "Build audit log viewer for admin users",
    "Add dark mode theme toggle",
    "Create multi-step registration wizard",
    "Implement activity feed for user actions",
    "Build bulk operations for data management",
    "Add keyboard shortcuts for power users",
    "Create data visualization charts",
    "Implement drag-and-drop file uploader",
    "Build calendar view for events",
    "Add rich text editor for content creation",
    "Create tag management system",
    "Implement advanced search filters",
    "Build export to PDF functionality",
    "Add social media sharing buttons",
    "Create comment system for posts",
    "Implement auto-save for forms",
    "Build notification badge counter",
    "Add image cropping and resize tool",
    "Create breadcrumb navigation",
    "Implement infinite scroll for feed",
    "Build quick actions menu",
    "Add tooltips and help text",
    "Create collapsible sidebar navigation",
    "Implement session timeout warning",
    "Build keyboard navigation support",
    "Add loading states and skeleton screens",
]

LOW_RISK_STORIES = [
    "Update footer copyright year",
    "Fix typo in welcome email template",
    "Change button color on homepage",
    "Update logo image in header",
    "Add new FAQ section to help page",
    "Update terms of service text",
    "Change default page title",
    "Add social media icons to footer",
    "Update contact email address",
    "Change placeholder text in login form",
    "Add new menu item to navigation",
    "Update privacy policy link",
    "Change error message text",
    "Add company address to footer",
    "Update success notification message",
    "Change icon for delete button",
    "Add alt text to images",
    "Update meta description for SEO",
    "Change font size in sidebar",
    "Add tooltip to help icon",
    "Update README documentation",
    "Change background color of banner",
    "Add margin to card components",
    "Update button text from 'Submit' to 'Save'",
    "Change order of menu items",
    "Add spacing between form fields",
    "Update placeholder in search box",
    "Change link color in footer",
    "Add new color to theme palette",
    "Update email signature template",
]

EPICS = [
    "User Authentication",
    "Payment Processing",
    "Admin Dashboard",
    "Reporting System",
    "Mobile App",
    "API Development",
    "Performance Optimization",
    "Security Enhancement",
    "UI/UX Improvements",
    "Data Management"
]

REPORTERS = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]


def generate_story_data(story_id: int, description: str, risk_level: str) -> dict:
    """Generate complete story data with realistic patterns based on risk level"""
    
    # Estimate story points based on risk and some randomness
    if risk_level == "High":
        estimated_points = random.choice([5, 8, 8, 13, 13])
    elif risk_level == "Medium":
        estimated_points = random.choice([2, 3, 3, 5, 5, 5])
    else:  # Low
        estimated_points = random.choice([1, 1, 2, 2, 3])
    
    # Generate actual points with risk-based underestimation patterns
    if risk_level == "High":
        # High risk stories often underestimated (50% chance of significant overrun)
        if random.random() < 0.5:
            actual_points = int(estimated_points * random.uniform(1.6, 2.5))
        else:
            actual_points = int(estimated_points * random.uniform(1.0, 1.4))
        caused_spillover = random.random() < 0.4  # 40% spillover rate
    elif risk_level == "Medium":
        # Medium risk sometimes underestimated
        if random.random() < 0.25:
            actual_points = int(estimated_points * random.uniform(1.3, 1.8))
        else:
            actual_points = int(estimated_points * random.uniform(0.9, 1.2))
        caused_spillover = random.random() < 0.15  # 15% spillover rate
    else:  # Low
        # Low risk usually accurate
        actual_points = int(estimated_points * random.uniform(0.8, 1.1))
        caused_spillover = random.random() < 0.05  # 5% spillover rate
    
    # Ensure minimum of 1 point
    actual_points = max(1, actual_points)
    
    # Days to complete (roughly based on points, with variation)
    days_to_complete = max(1, int(actual_points * random.uniform(0.8, 1.5)))
    
    return {
        "id": story_id,
        "description": description,
        "estimated_points": estimated_points,
        "actual_points": actual_points,
        "days_to_complete": days_to_complete,
        "caused_spillover": caused_spillover,
        "risk_level": risk_level,
        "epic": random.choice(EPICS),
        "reporter": random.choice(REPORTERS)
    }


def generate_seed_data():
    """Generate 150 stories with realistic risk distribution"""
    stories = []
    story_id = 1
    
    # High risk: 30 stories (20%)
    for description in random.sample(HIGH_RISK_STORIES, min(30, len(HIGH_RISK_STORIES))):
        stories.append(generate_story_data(story_id, description, "High"))
        story_id += 1
    
    # Medium risk: 75 stories (50%)
    medium_sample = random.choices(MEDIUM_RISK_STORIES, k=75)
    for description in medium_sample:
        stories.append(generate_story_data(story_id, description, "Medium"))
        story_id += 1
    
    # Low risk: 45 stories (30%)
    low_sample = random.choices(LOW_RISK_STORIES, k=45)
    for description in low_sample:
        stories.append(generate_story_data(story_id, description, "Low"))
        story_id += 1
    
    # Shuffle to mix risk levels (more realistic)
    random.shuffle(stories)
    
    # Re-assign sequential IDs after shuffle
    for idx, story in enumerate(stories, 1):
        story['id'] = idx
    
    return stories


def main():
    """Generate and save seed data"""
    print("Generating realistic seed data...")
    
    # Ensure data directory exists
    Path('data').mkdir(exist_ok=True)
    
    # Generate stories
    stories = generate_seed_data()
    
    # Save to JSON
    output_path = Path('data/seed_stories.json')
    with open(output_path, 'w') as f:
        json.dump(stories, f, indent=2)
    
    # Print statistics
    print(f"\n✓ Generated {len(stories)} stories")
    print(f"✓ Saved to {output_path}")
    
    risk_counts = {}
    spillover_count = 0
    underestimated_count = 0
    
    for story in stories:
        risk_counts[story['risk_level']] = risk_counts.get(story['risk_level'], 0) + 1
        if story['caused_spillover']:
            spillover_count += 1
        if story['actual_points'] > story['estimated_points'] * 1.5:
            underestimated_count += 1
    
    print("\nDataset Statistics:")
    print(f"  High Risk: {risk_counts.get('High', 0)} stories ({risk_counts.get('High', 0)/len(stories)*100:.1f}%)")
    print(f"  Medium Risk: {risk_counts.get('Medium', 0)} stories ({risk_counts.get('Medium', 0)/len(stories)*100:.1f}%)")
    print(f"  Low Risk: {risk_counts.get('Low', 0)} stories ({risk_counts.get('Low', 0)/len(stories)*100:.1f}%)")
    print(f"  Stories causing spillover: {spillover_count} ({spillover_count/len(stories)*100:.1f}%)")
    print(f"  Significantly underestimated: {underestimated_count} ({underestimated_count/len(stories)*100:.1f}%)")


if __name__ == '__main__':
    random.seed(42)  # For reproducibility
    main()

