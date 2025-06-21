from django.core.management.base import BaseCommand
from app.models import Patient  # Update 'app' to your actual app name

class Command(BaseCommand):
    help = 'Truncate all records in the Patient table'

    def handle(self, *args, **kwargs):
        count = Patient.objects.count()
        Patient.objects.all().delete()
        self.stdout.write(self.style.SUCCESS(f"Successfully deleted {count} records from Patient table."))
