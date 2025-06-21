import csv
import os
from django.core.management.base import BaseCommand
from app.models import Patient
from django.conf import settings

class Command(BaseCommand):
    help = 'Import patient data from CSV (only if table is empty)'

    def handle(self, *args, **kwargs):
        if Patient.objects.exists():
            self.stdout.write(self.style.WARNING("Data already exists. Aborting import."))
            return

        file_path = os.path.join(settings.MEDIA_ROOT, 'data', 'stroke.csv')

        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            patients = []
            for row in reader:
                try:
                    bmi = None if row['bmi'] in ['N/A', ''] else float(row['bmi'])

                    # Convert float strings to bool safely
                    hypertension = float(row['hypertension']) >= 1.0
                    heart_disease = float(row['heart_disease']) >= 1.0
                    stroke = float(row['stroke']) >= 1.0

                    patients.append(Patient(
                        gender=row['gender'],
                        age=int(float(row['age'])),  # Also fix for floats like 54.0
                        hypertension=hypertension,
                        heart_disease=heart_disease,
                        ever_married=row['ever_married'],
                        work_type=row['work_type'],
                        residence_type=row['Residence_type'],
                        avg_glucose_level=float(row['avg_glucose_level']),
                        bmi=bmi,
                        smoking_status=row['smoking_status'],
                        stroke=stroke
                    ))
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"Skipping row due to error: {e}"))

            Patient.objects.bulk_create(patients)
            self.stdout.write(self.style.SUCCESS(f"Imported {len(patients)} patient records."))
